"use client";

import { ThreadSummary } from "@/lib/types";
import styles from "./Sidebar.module.css";

interface SidebarProps {
  threads: ThreadSummary[];
  currentThreadId: string;
  disabled: boolean;
  onNewChat: () => void;
  onSwitchThread: (threadId: string) => void;
}

export default function Sidebar({
  threads,
  currentThreadId,
  disabled,
  onNewChat,
  onSwitchThread,
}: SidebarProps) {
  return (
    <aside className={styles.sidebar}>
      <button
        className={styles.newChat}
        onClick={onNewChat}
        disabled={disabled}
      >
        🧹 New chat
      </button>
      <div className={styles.divider} />
      <div className={styles.list}>
        {threads.map((thread) => {
          const isCurrent = thread.thread_id === currentThreadId;
          return (
            <button
              key={thread.thread_id}
              className={`${styles.threadButton} ${
                isCurrent ? styles.threadButtonCurrent : ""
              }`}
              onClick={() => onSwitchThread(thread.thread_id)}
              disabled={disabled || isCurrent}
              title={thread.preview || "(empty conversation)"}
            >
              {isCurrent ? "💬 " : ""}
              {thread.preview || "(empty conversation)"}
            </button>
          );
        })}
      </div>
    </aside>
  );
}
