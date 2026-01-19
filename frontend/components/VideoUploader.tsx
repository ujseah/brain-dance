// Brain Dance - Video Uploader Component
// TODO: Implement drag-and-drop upload with progress

"use client";

import { useState, useCallback } from "react";

interface VideoUploaderProps {
  onUploadComplete: (jobId: string) => void;
}

export function VideoUploader({ onUploadComplete }: VideoUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("video/")) {
        await uploadFile(file);
      }
    },
    [onUploadComplete]
  );

  const uploadFile = async (file: File) => {
    setUploading(true);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append("video", file);

      // TODO: Use XMLHttpRequest for progress tracking
      const response = await fetch("/api/video/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      onUploadComplete(data.job_id);
    } catch (error) {
      console.error("Upload failed:", error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div
      className={`
        border-2 border-dashed rounded-lg p-12 text-center transition-colors
        ${isDragging ? "border-blue-500 bg-blue-500/10" : "border-gray-600 hover:border-gray-500"}
      `}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {uploading ? (
        <div>
          <p className="text-lg mb-4">Uploading...</p>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      ) : (
        <div className="text-gray-400">
          <p className="text-lg">Drop your video here</p>
          <p className="text-sm mt-2">or click to browse</p>
        </div>
      )}
    </div>
  );
}
