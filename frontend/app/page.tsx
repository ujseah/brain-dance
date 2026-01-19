// Brain Dance - Video Upload Page
// TODO: Implement video upload with drag-and-drop

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-16">
        <h1 className="text-4xl font-bold text-center mb-4">
          Brain Dance
        </h1>
        <p className="text-xl text-gray-400 text-center mb-12">
          Transform video into explorable 3D worlds
        </p>

        {/* Video Upload Area */}
        <div className="max-w-2xl mx-auto">
          <div className="border-2 border-dashed border-gray-600 rounded-lg p-12 text-center hover:border-gray-500 transition-colors">
            <div className="text-gray-400 mb-4">
              <svg
                className="w-16 h-16 mx-auto mb-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <p className="text-lg">Drop your video here</p>
              <p className="text-sm mt-2">or click to browse</p>
            </div>
            <input
              type="file"
              accept="video/*"
              className="hidden"
              // TODO: Add onChange handler
            />
            <p className="text-xs text-gray-500 mt-4">
              Supports MP4, MOV, AVI, WebM (max 5 minutes)
            </p>
          </div>
        </div>

        {/* How it works */}
        <div className="mt-16 max-w-4xl mx-auto">
          <h2 className="text-2xl font-semibold text-center mb-8">
            How it works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[
              { step: 1, title: "Upload", desc: "Drop your video" },
              { step: 2, title: "Process", desc: "Extract frames & poses" },
              { step: 3, title: "Train", desc: "Build 3D Gaussians" },
              { step: 4, title: "Explore", desc: "Navigate your world" },
            ].map(({ step, title, desc }) => (
              <div key={step} className="text-center">
                <div className="w-12 h-12 rounded-full bg-blue-600 text-white flex items-center justify-center mx-auto mb-3 text-lg font-semibold">
                  {step}
                </div>
                <h3 className="font-medium">{title}</h3>
                <p className="text-sm text-gray-400">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
