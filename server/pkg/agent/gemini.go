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
			if line == "" {
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
			case "chunk":
				if msg.Delta != "" {
					output.WriteString(msg.Delta)
					trySend(msgCh, Message{Type: MessageText, Content: msg.Delta})
				}
			case "thought":
				if msg.Thought != "" {
					trySend(msgCh, Message{Type: MessageThinking, Content: msg.Thought})
				}
			case "call":
				if msg.Call != nil {
					trySend(msgCh, Message{
						Type:   MessageToolUse,
						Tool:   msg.Call.Name,
						CallID: msg.Call.ID,
						Input:  msg.Call.Input,
					})
				}
			case "response":
				trySend(msgCh, Message{
					Type:   MessageToolResult,
					CallID: msg.CallID,
					Output: msg.Output,
				})
			case "status":
				trySend(msgCh, Message{Type: MessageStatus, Status: msg.Status})
			case "info":
				trySend(msgCh, Message{Type: MessageLog, Level: "info", Content: msg.Message})
			case "result":
				if msg.Response != "" {
					output.Reset()
					output.WriteString(msg.Response)
				}
				if msg.Stats != nil {
					for modelName, stats := range msg.Stats.Models {
						u := usage[modelName]
						u.InputTokens += stats.Tokens.Input
						u.OutputTokens += stats.Tokens.Candidates
						u.CacheReadTokens += stats.Tokens.Cached
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

	// chunk fields
	Delta string `json:"delta,omitempty"`

	// thought fields
	Thought string `json:"thought,omitempty"`

	// call fields
	Call *geminiToolCall `json:"call,omitempty"`

	// response fields
	CallID string `json:"call_id,omitempty"`
	Output string `json:"output,omitempty"`

	// status/info fields
	Status  string `json:"status,omitempty"`
	Message string `json:"message,omitempty"`

	// result fields
	Response string       `json:"response,omitempty"`
	Stats    *geminiStats `json:"stats,omitempty"`
}

type geminiToolCall struct {
	ID    string         `json:"id"`
	Name  string         `json:"name"`
	Input map[string]any `json:"input"`
}

type geminiStats struct {
	Models map[string]geminiModelStats `json:"models"`
}

type geminiModelStats struct {
	Tokens struct {
		Input      int64 `json:"input"`
		Candidates int64 `json:"candidates"`
		Cached     int64 `json:"cached"`
	} `json:"tokens"`
}
