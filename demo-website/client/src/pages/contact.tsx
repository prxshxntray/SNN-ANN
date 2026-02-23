import NavBar from "@/components/NavBar";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { MapPin, Mail, Send } from "lucide-react";

export default function Contact() {
  const [submitted, setSubmitted] = useState(false);
  const [form, setForm] = useState({ name: "", email: "", company: "", message: "" });

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitted(true);
  }

  return (
    <div className="min-h-screen bg-[#0B0F14] text-white">
      <NavBar />
      <div className="pt-28 pb-24 px-6">
        <div className="max-w-4xl mx-auto">

          <div className="text-center mb-14">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-sm border border-[#70A0D0]/30 bg-[#70A0D0]/10 mb-4">
              <div className="h-1.5 w-1.5 rounded-full bg-[#70A0D0] animate-pulse" />
              <span className="text-[#70A0D0] text-xs font-mono tracking-widest">IN ACTIVE DEVELOPMENT</span>
            </div>
            <h1 className="text-4xl font-semibold text-white mb-4">Get in touch</h1>
            <p className="text-white/50 text-base max-w-lg mx-auto leading-relaxed">
              Wattr is currently working with select early-access partners.
              Tell us about your facility and we'll reach out.
            </p>
          </div>

          <div className="grid md:grid-cols-[1fr_360px] gap-8">
            <div className="rounded-md border border-white/8 bg-[#0d1520] p-8 relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-[#70A0D0]/30 to-transparent" />

              {submitted ? (
                <div className="flex flex-col items-center justify-center h-full py-12 text-center">
                  <div className="w-14 h-14 rounded-full bg-[#70A0D0]/20 border border-[#70A0D0]/40 flex items-center justify-center mb-5">
                    <Send className="w-6 h-6 text-[#70A0D0]" />
                  </div>
                  <h3 className="text-white text-xl font-semibold mb-2">Message received</h3>
                  <p className="text-white/50 text-sm max-w-xs">
                    Thank you for your interest. Our team will be in touch within 2 business days.
                  </p>
                  <button
                    onClick={() => { setSubmitted(false); setForm({ name: "", email: "", company: "", message: "" }); }}
                    className="mt-6 text-[#70A0D0] text-sm underline underline-offset-2"
                  >
                    Send another message
                  </button>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-5">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <Label htmlFor="name" className="text-white/60 text-xs font-mono uppercase tracking-widest">Full name</Label>
                      <Input
                        id="name"
                        data-testid="input-name"
                        required
                        placeholder="Alex Kim"
                        value={form.name}
                        onChange={(e) => setForm({ ...form, name: e.target.value })}
                        className="bg-[#070c12] border-white/10 text-white placeholder:text-white/20 focus-visible:ring-[#70A0D0]/40"
                      />
                    </div>
                    <div className="space-y-1.5">
                      <Label htmlFor="company" className="text-white/60 text-xs font-mono uppercase tracking-widest">Company</Label>
                      <Input
                        id="company"
                        data-testid="input-company"
                        placeholder="Acme Corp"
                        value={form.company}
                        onChange={(e) => setForm({ ...form, company: e.target.value })}
                        className="bg-[#070c12] border-white/10 text-white placeholder:text-white/20 focus-visible:ring-[#70A0D0]/40"
                      />
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="email" className="text-white/60 text-xs font-mono uppercase tracking-widest">Email</Label>
                    <Input
                      id="email"
                      type="email"
                      data-testid="input-email"
                      required
                      placeholder="alex@company.com"
                      value={form.email}
                      onChange={(e) => setForm({ ...form, email: e.target.value })}
                      className="bg-[#070c12] border-white/10 text-white placeholder:text-white/20 focus-visible:ring-[#70A0D0]/40"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="message" className="text-white/60 text-xs font-mono uppercase tracking-widest">Tell us about your facility</Label>
                    <Textarea
                      id="message"
                      data-testid="textarea-message"
                      required
                      placeholder="Facility size, current challenges, timeline..."
                      rows={5}
                      value={form.message}
                      onChange={(e) => setForm({ ...form, message: e.target.value })}
                      className="bg-[#070c12] border-white/10 text-white placeholder:text-white/20 focus-visible:ring-[#70A0D0]/40 resize-none"
                    />
                  </div>
                  <Button
                    type="submit"
                    data-testid="button-submit"
                    className="w-full bg-[#70A0D0] text-[#0B0F14] font-semibold"
                  >
                    <Send className="w-4 h-4 mr-2" />
                    Send message
                  </Button>
                </form>
              )}
            </div>

            <div className="space-y-4">
              <div className="rounded-md border border-white/8 bg-[#0d1520] p-5">
                <p className="text-[#70A0D0] text-xs font-mono tracking-widest uppercase mb-4">Status</p>
                <div className="space-y-3">
                  {[
                    { label: "Product phase", value: "Private Beta" },
                    { label: "Partners onboarded", value: "4 facilities" },
                    { label: "Target GA", value: "Q3 2026" },
                  ].map((item) => (
                    <div key={item.label} className="flex justify-between items-center">
                      <span className="text-white/40 text-xs">{item.label}</span>
                      <span className="text-white text-sm font-mono">{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-md border border-white/8 bg-[#0d1520] p-5">
                <p className="text-[#70A0D0] text-xs font-mono tracking-widest uppercase mb-4">Contact</p>
                <div className="space-y-3">
                  <div className="flex items-center gap-2.5">
                    <Mail className="w-3.5 h-3.5 text-white/30 flex-shrink-0" />
                    <span className="text-white/60 text-xs">prashant.rai@u.nus.edu</span>
                  </div>
                  <div className="flex items-center gap-2.5">
                    <MapPin className="w-3.5 h-3.5 text-white/30 flex-shrink-0" />
                    <span className="text-white/60 text-xs">London, United Kingdom</span>
                  </div>
                </div>
              </div>

              <div className="rounded-md border border-[#70A0D0]/20 bg-[#0a1624] p-5">
                <p className="text-[#70A0D0] text-xs font-mono tracking-widest uppercase mb-3">Early Access</p>
                <p className="text-white/50 text-xs leading-relaxed">
                  Early-access partners receive dedicated onboarding, custom model training on their estate data, 
                  and fixed early-adopter pricing locked for 3 years.
                </p>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {["Custom Training", "Priority Support", "Fixed Pricing"].map((tag) => (
                    <Badge key={tag} className="bg-[#70A0D0]/10 text-[#70A0D0] border-[#70A0D0]/20 text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
