import React from 'react';

const UploadForm = ({ onFileSelect, previewUrl, onPredict, isLoading }) => {
    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            onFileSelect(e.target.files[0]);
        }
    };

    return (
        <div className="w-full p-5 box-border rounded-10 shadow-[0_0_10px_rgb(97,91,91)] overflow-auto bg-[#a46161] animate-[slideInLeft_0.5s_ease-out]">
            <h1 className="text-[22px] mb-[13px] text-black text-center font-bold">
                Upload an Image of Wheat Plant
            </h1>

            <div className="flex flex-col items-center">
                <label htmlFor="file-upload" className="inline-block px-5 py-2.5 cursor-pointer rounded-md bg-[#007bff] text-white text-[16px] mb-[15px] transition-all duration-300 hover:bg-[#0056b3] hover:scale-105 hover:shadow-lg">
                    Choose File
                </label>
                <input
                    id="file-upload"
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                />

                {previewUrl && (
                    <div className="my-5 text-center animate-[fadeIn_1s_ease]">
                        <img
                            src={previewUrl}
                            alt="Image Preview"
                            className="max-w-[80%] rounded-lg border-2 border-[#ca2d2d] shadow-[0_0_15px_rgb(205,31,31)] mx-auto"
                        />
                    </div>
                )}

                <button
                    onClick={onPredict}
                    disabled={!previewUrl || isLoading}
                    className="bg-[#a82111] text-black px-[25px] py-[12px] border-none rounded-md cursor-pointer text-[18px] transition-all duration-300 hover:bg-[#867b7b] hover:scale-105 hover:shadow-[0_0_20px_rgb(255,240,240)] disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {isLoading ? 'Predicting...' : 'Predict'}
                </button>
            </div>
        </div>
    );
};

export default UploadForm;
