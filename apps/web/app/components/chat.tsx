"use client";

import { useRef, useState, useCallback } from "react";
import { SidebarTrigger } from "@multica/ui/components/ui/sidebar";
import { Badge } from "@multica/ui/components/ui/badge";
import { Button } from "@multica/ui/components/ui/button";
import { ChatInput } from "@multica/ui/components/chat-input";
import { MemoizedMarkdown } from "@multica/ui/components/markdown";
import { HugeiconsIcon } from "@hugeicons/react";
import { UserIcon, Copy01Icon, CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { toast } from "@multica/ui/components/ui/sonner";
import { useMessages } from "../hooks/use-messages";
import { useGateway } from "../hooks/use-gateway";
import { useHub } from "../hooks/use-hub";
import { useActiveAgent } from "../hooks/use-active-agent";
import { useDeviceId } from "../hooks/use-device-id";
import { useScrollFade } from "../hooks/use-scroll-fade";
import { cn } from "@multica/ui/lib/utils";

const STATE_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  registered: "default",
  connected: "secondary",
  connecting: "secondary",
  disconnected: "destructive",
}

export function Chat() {
  const activeAgentId = useActiveAgent((s) => s.activeAgentId)
  const { messages, addUserMessage, addAssistantMessage } = useMessages()
  const { hub } = useHub()

  const { state: gwState, send } = useGateway({
    onMessage: (msg) => {
      const payload = msg.payload as { agentId?: string; content?: string }
      if (payload?.agentId && payload?.content) {
        addAssistantMessage(payload.content, payload.agentId)
      }
    },
  })

  const handleSend = useCallback((text: string) => {
    if (!hub?.hubId || !activeAgentId) return
    addUserMessage(text, activeAgentId)
    send(hub.hubId, "message", { agentId: activeAgentId, content: text })
  }, [hub?.hubId, activeAgentId, addUserMessage, send])

  const filtered = activeAgentId
    ? messages.filter(m => m.agentId === activeAgentId)
    : []

  const canSend = gwState === "registered" && !!activeAgentId

  const deviceId = useDeviceId()
  const [deviceCopied, setDeviceCopied] = useState(false)
  const handleCopyDevice = useCallback(async () => {
    if (!deviceId) return
    await navigator.clipboard.writeText(deviceId)
    setDeviceCopied(true)
    toast.success("Device ID copied")
    setTimeout(() => setDeviceCopied(false), 2000)
  }, [deviceId])

  const mainRef = useRef<HTMLElement>(null)
  const fadeStyle = useScrollFade(mainRef)

  return (
    <div className="h-dvh flex flex-col overflow-hidden w-full">
      <header className="flex items-center gap-2 p-2">
        <SidebarTrigger />
        {deviceId && (
          <>
            <span className="text-xs text-muted-foreground font-mono">
              {deviceId}
            </span>
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={handleCopyDevice}
              aria-label="Copy device ID"
            >
              <HugeiconsIcon
                icon={deviceCopied ? CheckmarkCircle02Icon : Copy01Icon}
                strokeWidth={2}
                className={cn("size-3", deviceCopied && "text-green-500")}
              />
            </Button>
          </>
        )}
        <Badge variant={STATE_VARIANT[gwState] ?? "outline"} className="text-xs">
          {gwState}
        </Badge>
        {activeAgentId && (
          <span className="text-xs text-muted-foreground font-mono">
            Agent: {activeAgentId.slice(0, 8)}...
          </span>
        )}
      </header>

      <main ref={mainRef} className="flex-1 overflow-y-auto min-h-0" style={fadeStyle}>
        {!activeAgentId ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
            <HugeiconsIcon icon={UserIcon} strokeWidth={1.5} className="size-10 opacity-30" />
            <span className="text-sm">Select an agent to start chatting</span>
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
            Send a message to start the conversation
          </div>
        ) : (
          <div className="px-4 py-6 space-y-6 max-w-4xl mx-auto">
            {filtered.map((msg) => (
              <div
                key={msg.id}
                className={cn(
                  "flex",
                  msg.role === "user" ? "justify-end" : "justify-start"
                )}
              >
                <div
                  className={cn(
                    msg.role === "user" ? "bg-muted rounded-md max-w-[60%] p-1 px-2.5" : "w-full max-w-[90%] p-1 px-2.5"
                  )}
                >
                  <MemoizedMarkdown mode="minimal" id={msg.id}>
                    {msg.content}
                  </MemoizedMarkdown>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      <footer className="w-full p-2 pt-1 max-w-4xl mx-auto">
        <ChatInput
          onSubmit={handleSend}
          disabled={!canSend}
          placeholder={!activeAgentId ? "Select an agent first..." : "Type a message..."}
        />
      </footer>
    </div>
  );
}
