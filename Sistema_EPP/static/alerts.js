(() => {
  const root = document.getElementById("alert-root");
  let audioReady = false;
  let lastMessage = "";

  function beep() {
    if (!audioReady) return;
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.frequency.value = 880;
    gain.gain.value = 0.08;
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    setTimeout(() => {
      osc.stop();
      ctx.close();
    }, 180);
  }

  document.addEventListener("click", () => {
    audioReady = true;
  }, { once: true });

  async function refreshAlerts() {
    try {
      const response = await fetch("/estado.json", { cache: "no-store" });
      const data = await response.json();
      const message = (data.alertas || []).join(" | ");
      if (!message) {
        root.hidden = true;
        root.textContent = "";
        lastMessage = "";
        return;
      }
      root.hidden = false;
      root.textContent = "ALERTA DE EPP: " + message;
      if (message !== lastMessage) {
        beep();
        lastMessage = message;
      }
    } catch (_) {
      root.hidden = true;
    }
  }

  refreshAlerts();
  setInterval(refreshAlerts, 2000);
})();
