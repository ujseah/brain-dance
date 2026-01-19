// Brain Dance - Gaussian Splat Viewer Component
// TODO: Integrate Spark.js for rendering

"use client";

import { useEffect, useRef } from "react";

interface GaussianViewerProps {
  splatUrl: string;
  className?: string;
}

export function GaussianViewer({ splatUrl, className }: GaussianViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // TODO: Initialize Spark.js viewer
    // import('spark').then(({ Viewer }) => {
    //   const viewer = new Viewer({
    //     container: containerRef.current,
    //     url: splatUrl,
    //   });
    //
    //   viewer.on('load', () => {
    //     console.log('Scene loaded');
    //   });
    //
    //   return () => viewer.dispose();
    // });

    console.log("GaussianViewer: Would load", splatUrl);
  }, [splatUrl]);

  return (
    <div ref={containerRef} className={className}>
      {/* Placeholder */}
      <div className="w-full h-full flex items-center justify-center bg-gray-900 text-gray-500">
        <p>Spark.js viewer: {splatUrl}</p>
      </div>
    </div>
  );
}
