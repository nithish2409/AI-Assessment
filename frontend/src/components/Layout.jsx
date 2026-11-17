import { useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
    BrainCircuit,
    LayoutDashboard,
    SplitSquareVertical,
    GitCompareArrows,
    ShieldAlert,
    Newspaper,
    Activity,
    Menu,
    X,
    ExternalLink,
} from "lucide-react";
import SystemHealthIndicator from "./SystemHealthIndicator";

const navSections = [
    {
        title: "Main Dashboard",
        items: [
            { to: "/", label: "Dashboard Overview", icon: LayoutDashboard },
        ],
    },
    {
        title: "Computer Vision & Inspection",
        items: [
            { to: "/inspection", label: "AI Inspection & Severity", icon: ShieldAlert },
            { to: "/image-comparison", label: "Before/After Comparison", icon: SplitSquareVertical },
            { to: "/hybrid-analysis", label: "Hybrid Progression", icon: GitCompareArrows },
        ],
    },
    {
        title: "Natural Language Processing",
        items: [
            { to: "/news-classification", label: "News Classification", icon: Newspaper },
        ],
    },
    {
        title: "Diagnostics",
        items: [
            { to: "/system-status", label: "System Status", icon: Activity },
        ],
    },
];

export default function Layout() {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const location = useLocation();

    return (
        <div className="min-h-screen bg-slate-50 md:flex">
            {/* Mobile Top Bar */}
            <div className="flex items-center justify-between border-b border-slate-200 bg-slate-950 p-4 text-white md:hidden">
                <div className="flex items-center gap-2.5">
                    <div className="rounded-xl bg-blue-600 p-1.5 text-white">
                        <BrainCircuit size={20} />
                    </div>
                    <div>
                        <h1 className="text-sm font-bold leading-tight">AI Assessment</h1>
                        <p className="text-[10px] text-slate-400">Engineering Dashboard</p>
                    </div>
                </div>

                <button
                    type="button"
                    onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                    className="rounded-lg p-2 text-slate-400 hover:bg-slate-900 hover:text-white"
                    aria-label="Toggle navigation menu"
                >
                    {mobileMenuOpen ? <X size={22} /> : <Menu size={22} />}
                </button>
            </div>

            {/* Backdrop for Mobile Drawer */}
            {mobileMenuOpen && (
                <div
                    className="fixed inset-0 z-40 bg-slate-950/60 backdrop-blur-xs md:hidden"
                    onClick={() => setMobileMenuOpen(false)}
                />
            )}

            {/* Sidebar Navigation */}
            <aside
                className={`fixed inset-y-0 left-0 z-50 flex w-72 flex-col justify-between border-r border-slate-800 bg-slate-950 p-5 text-white transition-transform duration-200 ease-in-out md:static md:w-68 md:shrink-0 md:translate-x-0 ${
                    mobileMenuOpen ? "translate-x-0" : "-translate-x-full"
                }`}
            >
                <div>
                    {/* Header Branding */}
                    <div className="mb-6 flex items-center gap-3">
                        <div className="rounded-xl bg-blue-600 p-2 text-white shadow-md shadow-blue-500/20">
                            <BrainCircuit size={24} />
                        </div>
                        <div>
                            <h1 className="font-bold tracking-tight text-white text-base">
                                AI Assessment
                            </h1>
                            <p className="text-xs text-slate-400">
                                Engineering Dashboard
                            </p>
                        </div>
                    </div>

                    {/* Navigation Groups */}
                    <nav className="space-y-5 overflow-y-auto max-h-[calc(100vh-250px)] scrollbar-none pr-1">
                        {navSections.map((sec, idx) => (
                            <div key={idx}>
                                <p className="mb-2 px-3 text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                    {sec.title}
                                </p>
                                <div className="space-y-1">
                                    {sec.items.map(({ to, label, icon: Icon }) => (
                                        <NavLink
                                            key={to}
                                            to={to}
                                            end={to === "/"}
                                            onClick={() => setMobileMenuOpen(false)}
                                            className={({ isActive }) =>
                                                `flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-xs font-medium transition ${
                                                    isActive
                                                        ? "bg-blue-600 text-white shadow-sm"
                                                        : "text-slate-300 hover:bg-slate-900 hover:text-white"
                                                }`
                                            }
                                        >
                                            <Icon size={16} className="shrink-0" />
                                            <span>{label}</span>
                                        </NavLink>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </nav>
                </div>

                {/* System Health Strip in Sidebar */}
                <div className="mt-6 border-t border-slate-800 pt-4">
                    <SystemHealthIndicator />
                </div>
            </aside>

            {/* Main Application Content */}
            <main className="min-w-0 flex-1 p-4 md:p-8">
                <div className="mx-auto max-w-6xl">
                    <Outlet />
                </div>
            </main>
        </div>
    );
}
