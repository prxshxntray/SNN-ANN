import { Link, useLocation } from "wouter";
import wattrLogo from "@assets/image_1771779566568.png";

const navItems = [
  { href: "/", label: "Home" },
  { href: "/noc", label: "Operations" },
  { href: "/technology", label: "Technology" },
  { href: "/contact", label: "Contact" },
];

export default function NavBar() {
  const [location] = useLocation();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-3 border-b border-white/10 bg-[#0B0F14]/80 backdrop-blur-md">
      <Link href="/">
        <div className="flex items-center gap-2 cursor-pointer" data-testid="link-logo">
          <img src={wattrLogo} alt="Wattr" className="h-8 w-auto brightness-200" />
          <span className="text-white font-semibold text-lg tracking-tight" style={{ fontFamily: "var(--font-sans)" }}>
            Wattr
          </span>
        </div>
      </Link>

      <div className="flex items-center gap-1">
        {navItems.map((item) => {
          const active = item.href === "/" ? location === "/" : location.startsWith(item.href);
          return (
            <Link key={item.href} href={item.href}>
              <span
                data-testid={`link-nav-${item.label.toLowerCase()}`}
                className={`
                  px-4 py-2 rounded-md text-sm font-medium transition-colors cursor-pointer
                  ${active
                    ? "bg-[#70A0D0]/20 text-[#70A0D0]"
                    : "text-white/60 hover:text-white/90 hover:bg-white/5"
                  }
                `}
              >
                {item.label}
              </span>
            </Link>
          );
        })}
      </div>

      <div className="flex items-center gap-2">
        <div className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
        <span className="text-xs text-white/40 font-mono">LIVE</span>
      </div>
    </nav>
  );
}
