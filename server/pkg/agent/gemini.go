package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// geminiBackend implements Backend by spawning the Gemini CLI
// with --output-format stream-json.
type geminiBackend struct {
	cfg Config
}

func (b *geminiBackend) Execute(ctx context.Context, prompt string, opts ExecOptions) (*Session, error) {
	execPath := b.cfg.ExecutablePath
	if execPath == "" {
		execPath = "gemini"
	}
	if _, err := exec.LookPath(execPath); err != nil {
		return nil, fmt.Errorf("gemini executable not found at %q: %w", execPath, err)
	}

	timeout := opts.Timeout
	if timeout == 0 {
		timeout = 20 * time.Minute
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)

	args := []string{
		"--prompt", prompt,
		"--output-format", "stream-json",
		"--approval-mode", "yolo",
	}
	if opts.Model != "" {
		args = append(args, "--model", opts.Model)
	}

	cmd := exec.CommandContext(runCtx, execPath, args...)
	if opts.Cwd != "" {
		cmd.Dir = opts.Cwd
	}
	cmd.Env = buildEnv(b.cfg.Env)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("gemini stdout pipe: %w", err)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("gemini stdin pipe: %w", err)
	}
	cmd.Stderr = newLogWriter(b.cfg.Logger, "[gemini:stderr] ")

	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("start gemini: %w", err)
	}

	b.cfg.Logger.Info("gemini started", "pid", cmd.Process.Pid, "cwd", opts.Cwd, "model", opts.Model)

	msgCh := make(chan Message, 256)
	resCh := make(chan Result, 1)

	go func() {
		defer cancel()
		defer close(msgCh)
		defer close(resCh)
		defer stdin.Close()

		startTime := time.Now()
		var output strings.Builder
		var sessionID string
		finalStatus := "completed"
		var finalError string
		usage := make(map[string]TokenUsage)

		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || !strings.HasPrefix(line, "{") {
				continue
			}

			var msg geminiSDKMessage
			if err := json.Unmarshal([]byte(line), &msg); err != nil {
				continue
			}

			if msg.SessionID != "" {
				sessionID = msg.SessionID
			}

			switch msg.Type {
			case "message":
				if msg.Role == "assistant" && msg.Content != "" {
					output.WriteString(msg.Content)
					trySend(msgCh, Message{Type: MessageText, Content: msg.Content})
				}
			case "thought":
				if msg.Content != "" {
					trySend(msgCh, Message{Type: MessageThinking, Content: msg.Content})
				}
			case "tool_use":
				trySend(msgCh, Message{
					Type:   MessageToolUse,
					Tool:   msg.ToolName,
					CallID: msg.ToolID,
					Input:  msg.Parameters,
				})
			case "tool_result":
				trySend(msgCh, Message{
					Type:   MessageToolResult,
					CallID: msg.ToolID,
					Output: msg.Output,
				})
			case "status":
				trySend(msgCh, Message{Type: MessageStatus, Status: msg.Status})
			case "result":
				if msg.Status == "error" || msg.Status == "fail" {
					finalStatus = "failed"
				}
				// Use the final response if available (especially if streaming didn't happen)
				if msg.Response != "" && output.Len() == 0 {
					output.WriteString(msg.Response)
				}
				if msg.Stats != nil {
					for modelName, stats := range msg.Stats.Models {
						u := usage[modelName]
						u.InputTokens += stats.InputTokens
						u.OutputTokens += stats.OutputTokens
						u.CacheReadTokens += stats.Cached
						usage[modelName] = u
					}
				}
			}
		}

		// Wait for process exit
		exitErr := cmd.Wait()
		duration := time.Since(startTime)

		if runCtx.Err() == context.DeadlineExceeded {
			finalStatus = "timeout"
			finalError = fmt.Sprintf("gemini timed out after %s", timeout)
		} else if runCtx.Err() == context.Canceled {
			finalStatus = "aborted"
			finalError = "execution cancelled"
		} else if exitErr != nil && finalStatus == "completed" {
			finalStatus = "failed"
			finalError = fmt.Sprintf("gemini exited with error: %v", exitErr)
		}

		b.cfg.Logger.Info("gemini finished", "pid", cmd.Process.Pid, "status", finalStatus, "duration", duration.Round(time.Millisecond).String())

		resCh <- Result{
			Status:     finalStatus,
			Output:     output.String(),
			Error:      finalError,
			DurationMs: duration.Milliseconds(),
			SessionID:  sessionID,
			Usage:      usage,
		}
	}()

	return &Session{Messages: msgCh, Result: resCh}, nil
}

// ── Gemini SDK JSON types ──

type geminiSDKMessage struct {
	Type      string `json:"type"`
	SessionID string `json:"session_id,omitempty"`

	// message/thought fields
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
	Delta   bool   `json:"delta,omitempty"`

	// tool_use fields
	ToolName   string         `json:"tool_name,omitempty"`
	ToolID     string         `json:"tool_id,omitempty"`
	Parameters map[string]any `json:"parameters,omitempty"`

	// tool_result fields
	Output string `json:"output,omitempty"`

	// status fields
	Status string `json:"status,omitempty"`

	// result fields
	Response string       `json:"response,omitempty"`
	Stats    *geminiStats `json:"stats,omitempty"`
}

type geminiStats struct {
	Models map[string]geminiModelStats `json:"models"`
}

type geminiModelStats struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	Cached       int64 `json:"cached"`
}
