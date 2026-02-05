import { describe, it, expect } from "vitest";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { pruneToolResults, DEFAULT_TOOL_RESULT_PRUNING_SETTINGS } from "./tool-result-pruning.js";

// Helper to create a user message with tool result
function createToolResultMessage(
  toolName: string,
  content: string,
  toolUseId: string = "tool-123",
): AgentMessage {
  return {
    role: "user",
    content: [
      {
        type: "tool_result",
        tool_use_id: toolUseId,
        name: toolName,
        content: [{ type: "text", text: content }],
      },
    ],
  } as unknown as AgentMessage;
}

// Helper to create an assistant message
function createAssistantMessage(text: string): AgentMessage {
  return {
    role: "assistant",
    content: [{ type: "text", text }],
  } as unknown as AgentMessage;
}

// Helper to create a user message
function createUserMessage(text: string): AgentMessage {
  return {
    role: "user",
    content: text,
  } as unknown as AgentMessage;
}

describe("pruneToolResults", () => {
  it("returns unchanged if utilization is below softTrimRatio", () => {
    const messages = [
      createUserMessage("Hello"),
      createAssistantMessage("Hi there!"),
      createToolResultMessage("read", "Short content"),
    ];

    const result = pruneToolResults({
      messages,
      contextWindowTokens: 200_000, // Very large window
    });

    expect(result.changed).toBe(false);
    expect(result.messages).toBe(messages);
    expect(result.softTrimmed).toBe(0);
    expect(result.hardCleared).toBe(0);
  });

  it("soft trims large tool results", () => {
    // Create a message with a large tool result (5000 chars)
    const largeContent = "A".repeat(5000);
    const messages = [
      createUserMessage("Hello"),
      createAssistantMessage("Processing..."),
      createToolResultMessage("read", largeContent),
      createAssistantMessage("Done!"),
      createAssistantMessage("Follow up"),
      createAssistantMessage("Another one"),
      createAssistantMessage("Protected message"), // This is protected (keepLastAssistants=3)
    ];

    const result = pruneToolResults({
      messages,
      contextWindowTokens: 10_000, // Small window to trigger pruning
      settings: {
        softTrimRatio: 0.1, // Low threshold to ensure pruning
      },
    });

    expect(result.changed).toBe(true);
    expect(result.softTrimmed).toBe(1);

    // Check that the trimmed message contains head + tail
    const trimmedMsg = result.messages[2] as any;
    const trimmedText = trimmedMsg.content[0].content[0].text;
    expect(trimmedText).toContain("A".repeat(100)); // Should have some head content
    expect(trimmedText).toContain("..."); // Truncation marker
    expect(trimmedText).toContain("[Tool result trimmed:");
  });

  it("hard clears when utilization exceeds hardClearRatio", () => {
    // Create multiple messages with large tool results
    const largeContent = "X".repeat(10000);
    const messages = [
      createUserMessage("Start"),
      createAssistantMessage("Processing 1"),
      createToolResultMessage("read", largeContent, "tool-1"),
      createAssistantMessage("Processing 2"),
      createToolResultMessage("exec", largeContent, "tool-2"),
      createAssistantMessage("Processing 3"),
      createToolResultMessage("glob", largeContent, "tool-3"),
      createAssistantMessage("Done 1"), // Protected
      createAssistantMessage("Done 2"), // Protected
      createAssistantMessage("Done 3"), // Protected
    ];

    const result = pruneToolResults({
      messages,
      contextWindowTokens: 5_000, // Very small window
      settings: {
        softTrimRatio: 0.1,
        hardClearRatio: 0.2,
        minPrunableToolChars: 1000, // Lower threshold for test
        hardClear: {
          enabled: true,
          placeholder: "[Cleared]",
        },
      },
    });

    expect(result.changed).toBe(true);
    // Should have cleared at least some tool results
    expect(result.hardCleared).toBeGreaterThan(0);
    expect(result.charsSaved).toBeGreaterThan(0);
  });

  it("protects last N assistant messages", () => {
    const messages = [
      createUserMessage("Hello"),
      createAssistantMessage("First"),
      createToolResultMessage("read", "A".repeat(5000), "tool-1"), // Should be prunable
      createAssistantMessage("Second"), // Protected (keepLastAssistants=3)
      createToolResultMessage("read", "B".repeat(5000), "tool-2"), // In protected zone, should NOT be pruned
      createAssistantMessage("Third"), // Protected
      createAssistantMessage("Fourth"), // Protected
    ];

    const result = pruneToolResults({
      messages,
      contextWindowTokens: 5_000,
      settings: {
        softTrimRatio: 0.1,
        keepLastAssistants: 3,
      },
    });

    // The first tool result (before protected zone) may be pruned
    // But the second one (after "Second" assistant which is in protected zone) should not be
    if (result.changed) {
      // Check that tool-2 result is NOT modified (it's in protected zone)
      const tool2Msg = result.messages[4] as any;
      const tool2Content = tool2Msg.content[0].content[0].text;
      expect(tool2Content).toBe("B".repeat(5000)); // Unchanged
    }
  });

  it("never prunes before first user message", () => {
    const messages = [
      createAssistantMessage("Bootstrap read"), // Before first user message
      createToolResultMessage("read", "A".repeat(5000), "tool-1"), // Should NOT be pruned
      createUserMessage("Hello"), // First user message
      createAssistantMessage("Response"),
      createToolResultMessage("read", "B".repeat(5000), "tool-2"), // Can be pruned
      createAssistantMessage("Done 1"),
      createAssistantMessage("Done 2"),
      createAssistantMessage("Done 3"),
    ];

    const result = pruneToolResults({
      messages,
      contextWindowTokens: 5_000,
      settings: {
        softTrimRatio: 0.1,
      },
    });

    // The first tool result (before first user message) should NOT be modified
    const tool1Msg = result.messages[1] as any;
    const tool1Content = tool1Msg.content[0].content[0].text;
    expect(tool1Content).toBe("A".repeat(5000)); // Unchanged - bootstrap protection
  });

  it("respects tool deny list", () => {
    const messages = [
      createUserMessage("Hello"),
      createAssistantMessage("Processing"),
      createToolResultMessage("read", "A".repeat(5000), "tool-1"),
      createAssistantMessage("Done 1"),
      createAssistantMessage("Done 2"),
      createAssistantMessage("Done 3"),
    ];

    const result = pruneToolResults({
      messages,
      contextWindowTokens: 5_000,
      settings: {
        softTrimRatio: 0.1,
        tools: {
          deny: ["read"], // Don't prune read tool results
        },
      },
    });

    // read tool should not be pruned
    expect(result.changed).toBe(false);
  });

  it("respects tool allow list", () => {
    const messages = [
      createUserMessage("Hello"),
      createAssistantMessage("Processing"),
      createToolResultMessage("read", "A".repeat(5000), "tool-1"),
      createToolResultMessage("exec", "B".repeat(5000), "tool-2"),
      createAssistantMessage("Done 1"),
      createAssistantMessage("Done 2"),
      createAssistantMessage("Done 3"),
    ];

    const result = pruneToolResults({
      messages,
      contextWindowTokens: 5_000,
      settings: {
        softTrimRatio: 0.1,
        tools: {
          allow: ["exec"], // Only prune exec tool results
        },
      },
    });

    if (result.changed) {
      // read tool should not be pruned
      const tool1Msg = result.messages[2] as any;
      const tool1Content = tool1Msg.content[0].content[0].text;
      expect(tool1Content).toBe("A".repeat(5000)); // Unchanged
    }
  });

  it("skips tool results with images", () => {
    const messages = [
      createUserMessage("Hello"),
      createAssistantMessage("Processing"),
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "tool-1",
            name: "screenshot",
            content: [
              { type: "image", source: { type: "base64", data: "abc123" } },
              { type: "text", text: "A".repeat(5000) },
            ],
          },
        ],
      } as unknown as AgentMessage,
      createAssistantMessage("Done 1"),
      createAssistantMessage("Done 2"),
      createAssistantMessage("Done 3"),
    ];

    const result = pruneToolResults({
      messages,
      contextWindowTokens: 5_000,
      settings: {
        softTrimRatio: 0.1,
      },
    });

    // Image-containing tool result should not be pruned
    expect(result.softTrimmed).toBe(0);
    expect(result.hardCleared).toBe(0);
  });
});

describe("DEFAULT_TOOL_RESULT_PRUNING_SETTINGS", () => {
  it("has expected default values", () => {
    expect(DEFAULT_TOOL_RESULT_PRUNING_SETTINGS.softTrimRatio).toBe(0.3);
    expect(DEFAULT_TOOL_RESULT_PRUNING_SETTINGS.hardClearRatio).toBe(0.5);
    expect(DEFAULT_TOOL_RESULT_PRUNING_SETTINGS.keepLastAssistants).toBe(3);
    expect(DEFAULT_TOOL_RESULT_PRUNING_SETTINGS.softTrim.maxChars).toBe(4000);
    expect(DEFAULT_TOOL_RESULT_PRUNING_SETTINGS.softTrim.headChars).toBe(1500);
    expect(DEFAULT_TOOL_RESULT_PRUNING_SETTINGS.softTrim.tailChars).toBe(1500);
    expect(DEFAULT_TOOL_RESULT_PRUNING_SETTINGS.hardClear.enabled).toBe(true);
  });
});
