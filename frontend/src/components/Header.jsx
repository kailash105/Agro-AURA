import React from 'react';

const Header = () => {
    return (
        <header className="w-full flex justify-between items-center p-5 bg-[#3a588d] text-white fixed top-0 z-50 border-b-2 border-[#e0e0e0]">
            <div className="flex items-center gap-5">
                <img src="/images/Agro-AURA.png" className="h-[60px] w-auto" alt="Agro-AURA Logo" />
                <div className="flex flex-col">
                    <div className="text-[32px] font-bold text-[#4CAF50] font-serif m-0">Agro-AURA</div>
                    <div className="text-[18px] text-[#ddd] m-0">Healthy Wheat Starts with Early Detection</div>
                </div>
            </div>
            <nav>
                <ul className="flex gap-5 m-0 p-0 list-none">
                    <li>
                        <a
                            href="https://forms.gle/z2UnM9VFiEEpBkQc6"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[#cecece] font-bold px-4 py-2 rounded transition-all duration-300 hover:bg-[#55cc00] hover:text-[#490000] hover:shadow-[inset_0_0_0_2em_#55cc00]"
                        >
                            Give us a Feedback
                        </a>
                    </li>
                </ul>
            </nav>
        </header>
    );
};

export default Header;
