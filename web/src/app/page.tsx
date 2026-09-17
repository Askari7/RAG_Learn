import { Suspense } from "react";
import ChatApp from "@/components/ChatApp";

export default function Page() {
  return (
    <Suspense fallback={null}>
      <ChatApp />
    </Suspense>
  );
}
