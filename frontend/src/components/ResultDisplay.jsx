import React from 'react';

const ResultDisplay = ({ prediction, cause, pesticide }) => {
    return (
        <>
            <div className="mt-5 p-5 rounded-lg border border-[#f5c6cb] bg-[#47ca8d] text-black shadow-sm animate-[slideInRight_0.5s_ease-out]">
                <h2 className="text-xl font-bold m-0">
                    Prediction: <span>{prediction}</span>
                </h2>
            </div>

            <div className="mt-5 p-5 rounded-lg border border-[#c3e6cb] bg-[#9dddac] text-black shadow-sm animate-[slideInRight_0.5s_ease-out]">
                <h2 className="text-xl font-bold m-0 mb-2">
                    Cause: <span>{cause}</span>
                </h2>
                <h2 className="text-xl font-bold m-0">
                    Recommended Pesticide: <span>{pesticide}</span>
                </h2>
            </div>
        </>
    );
};

export default ResultDisplay;
