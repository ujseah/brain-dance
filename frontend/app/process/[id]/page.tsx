// Brain Dance - Processing Progress Page
// TODO: Implement progress tracking with polling

interface Props {
  params: { id: string };
}

export default function ProcessingPage({ params }: Props) {
  const jobId = params.id;

  // TODO: Poll /api/video/{jobId}/status for progress updates

  return (
    <main className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
      <div className="text-center max-w-md">
        <h1 className="text-2xl font-bold mb-4">Processing Video</h1>
        <p className="text-gray-400 mb-8">Job ID: {jobId}</p>

        {/* Progress Bar */}
        <div className="w-full bg-gray-700 rounded-full h-3 mb-4">
          <div
            className="bg-blue-600 h-3 rounded-full transition-all duration-500"
            style={{ width: "45%" }}
          />
        </div>

        {/* Stage Info */}
        <div className="space-y-2 text-left bg-gray-800 rounded-lg p-4">
          <div className="flex justify-between">
            <span className="text-gray-400">Stage</span>
            <span>3DGS Training</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Progress</span>
            <span>45%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Iteration</span>
            <span>13,500 / 30,000</span>
          </div>
        </div>

        {/* Stage Timeline */}
        <div className="mt-8 flex justify-between text-sm">
          <div className="text-green-500">1. Extract</div>
          <div className="text-green-500">2. Poses</div>
          <div className="text-blue-500">3. Train</div>
          <div className="text-gray-500">4. Export</div>
        </div>
      </div>
    </main>
  );
}
