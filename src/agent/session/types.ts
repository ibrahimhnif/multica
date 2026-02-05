import type { AgentMessage } from "@mariozechner/pi-agent-core";

export type SessionMeta = {
  provider?: string;
  model?: string;
  thinkingLevel?: string;
  /** Reasoning mode: off, on, stream */
  reasoningMode?: string;
  /** Context window token 数 */
  contextWindowTokens?: number;
};

export type SessionEntry =
  | { type: "message"; message: AgentMessage; timestamp: number }
  | { type: "meta"; meta: SessionMeta; timestamp: number }
  | {
      type: "compaction";
      removed: number;
      kept: number;
      timestamp: number;
      /** Token 感知 compaction 信息（可选，向后兼容） */
      tokensRemoved?: number | undefined;
      tokensKept?: number | undefined;
      /** 摘要模式生成的摘要 */
      summary?: string | undefined;
      reason?: "count" | "tokens" | "summary" | "pruning" | undefined;
    };
