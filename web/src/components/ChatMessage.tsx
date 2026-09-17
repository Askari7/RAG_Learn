import ReactMarkdown from "react-markdown";
import { Message } from "@/lib/types";
import styles from "./ChatMessage.module.css";

export default function ChatMessage({ role, content }: Message) {
  return (
    <div className={`${styles.row} ${role === "user" ? styles.rowUser : ""}`}>
      <div
        className={`${styles.bubble} ${
          role === "user" ? styles.bubbleUser : styles.bubbleAssistant
        }`}
      >
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    </div>
  );
}
