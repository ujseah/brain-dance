// Brain Dance - 3DGS Viewer Page
// TODO: Integrate Spark.js for Gaussian Splat rendering

"use client";

import { useEffect, useRef } from "react";

interface Props {
  params: { id: string };
}

export default function ViewerPage({ params }: Props) {
  const jobId = params.id;
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // TODO: Load Spark.js viewer
    // const viewer = new Spark.Viewer({
    //   container: containerRef.current,
    //   url: `/api/video/${jobId}/scene.spz`,
    // });

    console.log("Viewer initialized for job:", jobId);

    return () => {
      // Cleanup viewer
    };
  }, [jobId]);

  return (
    <main className="h-screen bg-black">
      {/* Viewer Container */}
      <div ref={containerRef} className="w-full h-full">
        {/* Placeholder until Spark.js is integrated */}
        <div className="w-full h-full flex items-center justify-center text-white">
          <div className="text-center">
            <p className="text-xl mb-2">3D Gaussian Splat Viewer</p>
            <p className="text-gray-400">Job: {jobId}</p>
            <p className="text-sm text-gray-500 mt-4">
              Spark.js integration pending
            </p>
          </div>
        </div>
      </div>

      {/* Controls Overlay */}
      <div className="fixed bottom-4 left-4 right-4 flex justify-between items-center">
        <div className="bg-black/50 backdrop-blur rounded-lg px-4 py-2 text-white text-sm">
          <span className="text-gray-400">Controls:</span> Drag to orbit, Scroll
          to zoom, Shift+Drag to pan
        </div>
        <div className="flex gap-2">
          <button className="bg-white/10 hover:bg-white/20 backdrop-blur rounded-lg px-4 py-2 text-white text-sm transition-colors">
            Download
          </button>
          <button className="bg-white/10 hover:bg-white/20 backdrop-blur rounded-lg px-4 py-2 text-white text-sm transition-colors">
            Share
          </button>
        </div>
      </div>
    </main>
  );
}
