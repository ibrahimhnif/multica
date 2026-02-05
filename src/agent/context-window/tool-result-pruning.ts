/**
 * Tool Result Pruning
 *
 * Smart pruning of tool results to reduce context window usage while preserving
 * useful information. Implements two-phase pruning:
 *
 * 1. Soft Trim: Keep head + tail of large tool results
 * 2. Hard Clear: Replace old tool results with placeholder
 *
 * Based on OpenClaw's microcompact-style context pruning.
 */

import type { AgentMessage } from "@mariozechner/pi-agent-core";

// ─── Types ───────────────────────────────────────────────────────────────────

export type ToolResultPruningSettings = {
  /** Utilization ratio to start soft trimming (default: 0.3) */
  softTrimRatio: number;
  /** Utilization ratio to start hard clearing (default: 0.5) */
  hardClearRatio: number;
  /** Minimum prunable tool result chars to consider hard clear (default: 50000) */
  minPrunableToolChars: number;
  /** Number of recent assistant messages to protect from pruning (default: 3) */
  keepLastAssistants: number;
  /** Soft trim settings */
  softTrim: {
    /** Max chars before triggering soft trim (default: 4000) */
    maxChars: number;
    /** Chars to keep from start (default: 1500) */
    headChars: number;
    /** Chars to keep from end (default: 1500) */
    tailChars: number;
  };
  /** Hard clear settings */
  hardClear: {
    /** Whether hard clear is enabled (default: true) */
    enabled: boolean;
    /** Placeholder text for cleared results */
    placeholder: string;
  };
  /** Tool names to allow/deny pruning */
  tools?: {
    allow?: string[];
    deny?: string[];
  };
};

export const DEFAULT_TOOL_RESULT_PRUNING_SETTINGS: ToolResultPruningSettings = {
  softTrimRatio: 0.3,
  hardClearRatio: 0.5,
  minPrunableToolChars: 50_000,
  keepLastAssistants: 3,
  softTrim: {
    maxChars: 4_000,
    headChars: 1_500,
    tailChars: 1_500,
  },
  hardClear: {
    enabled: true,
    placeholder: "[Tool result cleared to save context space]",
  },
};

export type ToolResultPruningResult = {
  /** Pruned messages */
  messages: AgentMessage[];
  /** Whether any changes were made */
  changed: boolean;
  /** Number of soft-trimmed results */
  softTrimmed: number;
  /** Number of hard-cleared results */
  hardCleared: number;
  /** Estimated chars saved */
  charsSaved: number;
};

// ─── Constants ───────────────────────────────────────────────────────────────

const CHARS_PER_TOKEN_ESTIMATE = 4;
const IMAGE_CHAR_ESTIMATE = 8_000;

// ─── Helper Functions ────────────────────────────────────────────────────────

/**
 * Extract text content from a tool result content block.
 */
function extractToolResultText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      if (typeof block === "string") {
        parts.push(block);
      } else if (block && typeof block === "object") {
        if ("text" in block && typeof block.text === "string") {
          parts.push(block.text);
        }
      }
    }
    return parts.join("\n");
  }
  return "";
}

/**
 * Check if content contains images.
 */
function hasImageContent(content: unknown): boolean {
  if (!Array.isArray(content)) return false;
  for (const block of content) {
    if (block && typeof block === "object" && "type" in block) {
      if (block.type === "image") return true;
    }
  }
  return false;
}

/**
 * Estimate character count for a message.
 */
function estimateMessageChars(message: AgentMessage): number {
  const msgAny = message as any;

  if (message.role === "user") {
    const content = msgAny.content;
    if (typeof content === "string") return content.length;
    if (!Array.isArray(content)) return 0;

    let chars = 0;
    for (const block of content) {
      if (typeof block === "string") {
        chars += block.length;
      } else if (block && typeof block === "object") {
        if (block.type === "text" && typeof block.text === "string") {
          chars += block.text.length;
        } else if (block.type === "tool_result") {
          chars += extractToolResultText(block.content).length;
        } else if (block.type === "image") {
          chars += IMAGE_CHAR_ESTIMATE;
        }
      }
    }
    return chars;
  }

  if (message.role === "assistant") {
    const content = msgAny.content;
    if (typeof content === "string") return content.length;
    if (!Array.isArray(content)) return 0;

    let chars = 0;
    for (const block of content) {
      if (typeof block === "string") {
        chars += block.length;
      } else if (block && typeof block === "object") {
        if (block.type === "text" && typeof block.text === "string") {
          chars += block.text.length;
        } else if (block.type === "thinking" && typeof block.thinking === "string") {
          chars += block.thinking.length;
        } else if (block.type === "toolCall" || block.type === "tool_use") {
          try {
            chars += JSON.stringify(block.arguments ?? block.input ?? {}).length;
          } catch {
            chars += 128;
          }
        }
      }
    }
    return chars;
  }

  return 256;
}

/**
 * Estimate total character count for messages.
 */
function estimateContextChars(messages: AgentMessage[]): number {
  return messages.reduce((sum, m) => sum + estimateMessageChars(m), 0);
}

/**
 * Find the index where we should stop protecting assistant messages.
 * Returns null if not enough assistant messages exist.
 */
function findAssistantCutoffIndex(
  messages: AgentMessage[],
  keepLastAssistants: number,
): number | null {
  if (keepLastAssistants <= 0) return messages.length;

  let remaining = keepLastAssistants;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role !== "assistant") continue;
    remaining--;
    if (remaining === 0) return i;
  }

  return null;
}

/**
 * Check if a user message is a "real" user message (not just tool results).
 * Tool results are sent as user messages but they're not real user input.
 */
function isRealUserMessage(message: AgentMessage): boolean {
  if (message.role !== "user") return false;

  const msgAny = message as any;
  const content = msgAny.content;

  // String content is a real user message
  if (typeof content === "string") return true;

  // Array content - check if it has any non-tool-result blocks
  if (Array.isArray(content)) {
    for (const block of content) {
      if (typeof block === "string") return true;
      if (block && typeof block === "object") {
        // Any type other than tool_result is real user content
        if (block.type !== "tool_result") return true;
      }
    }
    // Only tool_result blocks - not a real user message
    return false;
  }

  return true;
}

/**
 * Find the index of the first real user message (not tool results).
 * This is used for bootstrap protection - we never prune before the first real user input.
 */
function findFirstUserIndex(messages: AgentMessage[]): number | null {
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg && isRealUserMessage(msg)) return i;
  }
  return null;
}

/**
 * Check if a tool should be pruned based on settings.
 */
function isToolPrunable(toolName: string, settings: ToolResultPruningSettings): boolean {
  const { tools } = settings;
  if (!tools) return true;

  // If deny list exists and tool is in it, don't prune
  if (tools.deny?.includes(toolName)) return false;

  // If allow list exists, only prune if tool is in it
  if (tools.allow && tools.allow.length > 0) {
    return tools.allow.includes(toolName);
  }

  return true;
}

/**
 * Take first N characters from text.
 */
function takeHead(text: string, maxChars: number): string {
  if (maxChars <= 0) return "";
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars);
}

/**
 * Take last N characters from text.
 */
function takeTail(text: string, maxChars: number): string {
  if (maxChars <= 0) return "";
  if (text.length <= maxChars) return text;
  return text.slice(text.length - maxChars);
}

/**
 * Soft trim a tool result text.
 */
function softTrimText(
  text: string,
  settings: ToolResultPruningSettings,
): { trimmed: string; saved: number } | null {
  const { maxChars, headChars, tailChars } = settings.softTrim;

  if (text.length <= maxChars) return null;
  if (headChars + tailChars >= text.length) return null;

  const head = takeHead(text, headChars);
  const tail = takeTail(text, tailChars);
  const note = `\n\n[Tool result trimmed: kept first ${headChars} chars and last ${tailChars} chars of ${text.length} chars.]`;
  const trimmed = `${head}\n...\n${tail}${note}`;

  return {
    trimmed,
    saved: text.length - trimmed.length,
  };
}

/**
 * Process a user message containing tool results.
 * Returns modified message if any tool results were trimmed/cleared.
 */
function processUserMessageToolResults(
  message: AgentMessage,
  settings: ToolResultPruningSettings,
  mode: "soft" | "hard",
): { message: AgentMessage; changed: boolean; charsSaved: number } {
  const msgAny = message as any;
  const content = msgAny.content;

  if (!Array.isArray(content)) {
    return { message, changed: false, charsSaved: 0 };
  }

  let changed = false;
  let charsSaved = 0;
  const newContent: any[] = [];

  for (const block of content) {
    if (!block || typeof block !== "object" || block.type !== "tool_result") {
      newContent.push(block);
      continue;
    }

    const toolName = block.name ?? "unknown";

    // Skip non-prunable tools
    if (!isToolPrunable(toolName, settings)) {
      newContent.push(block);
      continue;
    }

    // Skip image-containing tool results
    if (hasImageContent(block.content)) {
      newContent.push(block);
      continue;
    }

    const originalText = extractToolResultText(block.content);

    if (mode === "soft") {
      const result = softTrimText(originalText, settings);
      if (result) {
        newContent.push({
          ...block,
          content: [{ type: "text", text: result.trimmed }],
        });
        changed = true;
        charsSaved += result.saved;
      } else {
        newContent.push(block);
      }
    } else {
      // Hard clear
      newContent.push({
        ...block,
        content: [{ type: "text", text: settings.hardClear.placeholder }],
      });
      changed = true;
      charsSaved += originalText.length - settings.hardClear.placeholder.length;
    }
  }

  if (!changed) {
    return { message, changed: false, charsSaved: 0 };
  }

  return {
    message: { ...message, content: newContent } as AgentMessage,
    changed: true,
    charsSaved,
  };
}

// ─── Main Functions ──────────────────────────────────────────────────────────

/**
 * Prune tool results in messages to reduce context window usage.
 *
 * Two-phase approach:
 * 1. Soft Trim (at softTrimRatio): Keep head + tail of large tool results
 * 2. Hard Clear (at hardClearRatio): Replace old tool results with placeholder
 *
 * Protections:
 * - Never prunes before first user message (protects bootstrap/identity reads)
 * - Protects last N assistant messages and their corresponding tool results
 * - Skips image-containing tool results
 * - Respects tool allow/deny lists
 */
export function pruneToolResults(params: {
  messages: AgentMessage[];
  contextWindowTokens: number;
  settings?: Partial<ToolResultPruningSettings>;
}): ToolResultPruningResult {
  const { messages, contextWindowTokens } = params;
  const settings: ToolResultPruningSettings = {
    ...DEFAULT_TOOL_RESULT_PRUNING_SETTINGS,
    ...params.settings,
    softTrim: {
      ...DEFAULT_TOOL_RESULT_PRUNING_SETTINGS.softTrim,
      ...params.settings?.softTrim,
    },
    hardClear: {
      ...DEFAULT_TOOL_RESULT_PRUNING_SETTINGS.hardClear,
      ...params.settings?.hardClear,
    },
  };

  const charWindow = contextWindowTokens * CHARS_PER_TOKEN_ESTIMATE;
  if (charWindow <= 0) {
    return { messages, changed: false, softTrimmed: 0, hardCleared: 0, charsSaved: 0 };
  }

  // Find cutoff index for protected assistant messages
  const cutoffIndex = findAssistantCutoffIndex(messages, settings.keepLastAssistants);
  if (cutoffIndex === null) {
    return { messages, changed: false, softTrimmed: 0, hardCleared: 0, charsSaved: 0 };
  }

  // Never prune before first user message (bootstrap protection)
  const firstUserIndex = findFirstUserIndex(messages);
  const pruneStartIndex = firstUserIndex === null ? messages.length : firstUserIndex;

  // Calculate current utilization
  let totalChars = estimateContextChars(messages);
  let ratio = totalChars / charWindow;

  // No pruning needed
  if (ratio < settings.softTrimRatio) {
    return { messages, changed: false, softTrimmed: 0, hardCleared: 0, charsSaved: 0 };
  }

  let result = messages.slice();
  let changed = false;
  let softTrimmed = 0;
  let hardCleared = 0;
  let charsSaved = 0;

  // Track which messages have prunable tool results
  const prunableIndexes: number[] = [];

  // Phase 1: Soft Trim
  for (let i = pruneStartIndex; i < cutoffIndex; i++) {
    const msg = result[i];
    if (!msg || msg.role !== "user") continue;

    const msgAny = msg as any;
    if (!Array.isArray(msgAny.content)) continue;

    // Check if this message has tool results
    const hasToolResult = msgAny.content.some(
      (b: any) => b && typeof b === "object" && b.type === "tool_result",
    );
    if (!hasToolResult) continue;

    prunableIndexes.push(i);

    const processed = processUserMessageToolResults(msg, settings, "soft");
    if (processed.changed) {
      result[i] = processed.message;
      changed = true;
      softTrimmed++;
      charsSaved += processed.charsSaved;
      totalChars -= processed.charsSaved;
    }
  }

  // Recalculate ratio after soft trim
  ratio = totalChars / charWindow;

  // Phase 2: Hard Clear (if needed)
  if (ratio >= settings.hardClearRatio && settings.hardClear.enabled) {
    // Check if we have enough prunable content to make hard clear worthwhile
    let prunableChars = 0;
    for (const i of prunableIndexes) {
      prunableChars += estimateMessageChars(result[i]!);
    }

    if (prunableChars >= settings.minPrunableToolChars) {
      for (const i of prunableIndexes) {
        if (ratio < settings.hardClearRatio) break;

        const msg = result[i]!;
        const beforeChars = estimateMessageChars(msg);

        const processed = processUserMessageToolResults(msg, settings, "hard");
        if (processed.changed) {
          result[i] = processed.message;
          changed = true;
          hardCleared++;
          charsSaved += processed.charsSaved;
          totalChars -= processed.charsSaved;
          ratio = totalChars / charWindow;
        }
      }
    }
  }

  return {
    messages: result,
    changed,
    softTrimmed,
    hardCleared,
    charsSaved,
  };
}
