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

// copilotBackend implements Backend by spawning the Copilot CLI
// with --output-format stream-json.
type copilotBackend struct {
	cfg Config
}

func (b *copilotBackend) Execute(ctx context.Context, prompt string, opts ExecOptions) (*Session, error) {
	execPath := b.cfg.ExecutablePath
	if execPath == "" {
		execPath = "copilot"
	}
	if _, err := exec.LookPath(execPath); err != nil {
		return nil, fmt.Errorf("copilot executable not found at %q: %w", execPath, err)
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
		return nil, fmt.Errorf("copilot stdout pipe: %w", err)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("copilot stdin pipe: %w", err)
	}
	cmd.Stderr = newLogWriter(b.cfg.Logger, "[copilot:stderr] ")

	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("start copilot: %w", err)
	}

	b.cfg.Logger.Info("copilot started", "pid", cmd.Process.Pid, "cwd", opts.Cwd, "model", opts.Model)

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

			var msg geminiSDKMessage // Copilot CLI uses the same streaming format
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
			finalError = fmt.Sprintf("copilot timed out after %s", timeout)
		} else if runCtx.Err() == context.Canceled {
			finalStatus = "aborted"
			finalError = "execution cancelled"
		} else if exitErr != nil && finalStatus == "completed" {
			finalStatus = "failed"
			finalError = fmt.Sprintf("copilot exited with error: %v", exitErr)
		}

		b.cfg.Logger.Info("copilot finished", "pid", cmd.Process.Pid, "status", finalStatus, "duration", duration.Round(time.Millisecond).String())

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
