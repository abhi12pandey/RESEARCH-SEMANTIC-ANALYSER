/* ---------- Hero typewriter effect ---------- */
(function(){
  const phrases = [
    "Research insights, made simple.",
    "Upload. Ingest. Understand.",
    "Semantic analysis for your papers."
  ];
  const typedEl = document.getElementById("typed-text");
  const cursorEl = document.querySelector(".cursor");
  if(!typedEl) return;

  const typingSpeed = 40;   // ms per char
  const pauseBetween = 1400; // ms between phrases
  let phraseIndex = 0;
  let charIndex = 0;
  let deleting = false;

  function typeLoop(){
    const current = phrases[phraseIndex];
    if(!deleting){
      typedEl.textContent = current.slice(0, ++charIndex);
      if(charIndex === current.length){
        deleting = true;
        setTimeout(typeLoop, pauseBetween);
      } else {
        setTimeout(typeLoop, typingSpeed);
      }
    } else {
      typedEl.textContent = current.slice(0, --charIndex);
      if(charIndex === 0){
        deleting = false;
        phraseIndex = (phraseIndex + 1) % phrases.length;
        setTimeout(typeLoop, 200);
      } else {
        setTimeout(typeLoop, typingSpeed / 1.2);
      }
    }
  }
  // start slightly after load so other animations show
  setTimeout(typeLoop, 500);
  // show cursor
  if(cursorEl) cursorEl.style.opacity = 1;
})();
/* ---------- Hero typewriter effect ---------- */
(function(){
  const phrases = [
    "Research insights, made simple.",
    "Upload. Ingest. Understand.",
    "Semantic analysis for your papers."
  ];
  const typedEl = document.getElementById("typed-text");
  const cursorEl = document.querySelector(".cursor");
  if(!typedEl) return;

  const typingSpeed = 40;   // ms per char
  const pauseBetween = 1400; // ms between phrases
  let phraseIndex = 0;
  let charIndex = 0;
  let deleting = false;

  function typeLoop(){
    const current = phrases[phraseIndex];
    if(!deleting){
      typedEl.textContent = current.slice(0, ++charIndex);
      if(charIndex === current.length){
        deleting = true;
        setTimeout(typeLoop, pauseBetween);
      } else {
        setTimeout(typeLoop, typingSpeed);
      }
    } else {
      typedEl.textContent = current.slice(0, --charIndex);
      if(charIndex === 0){
        deleting = false;
        phraseIndex = (phraseIndex + 1) % phrases.length;
        setTimeout(typeLoop, 200);
      } else {
        setTimeout(typeLoop, typingSpeed / 1.2);
      }
    }
  }
  // start slightly after load so other animations show
  setTimeout(typeLoop, 500);
  // show cursor
  if(cursorEl) cursorEl.style.opacity = 1;
})();
// NAV toggle for mobile
document.addEventListener("DOMContentLoaded", function() {
  const navToggle = document.getElementById("nav-toggle");
  const navLinks = document.getElementById("nav-links");

  if (navToggle && navLinks) {
    navToggle.addEventListener("click", function() {
      const isShown = navLinks.classList.toggle("show");
      navToggle.setAttribute("aria-expanded", isShown ? "true" : "false");
    });
  }

  // Existing typewriter logic — keep it if present
  (function(){
    const typedEl = document.getElementById("typed-text");
    const cursorEl = document.querySelector(".cursor");
    if(!typedEl) return;
    const phrases = ["Research insights, made simple.", "Upload. Ingest. Understand.", "Semantic analysis for your papers."];
    let p=0, i=0, del=false;
    const speed = 40, pause=1400;
    function loop(){
      const cur = phrases[p];
      if(!del){
        typedEl.textContent = cur.slice(0, ++i);
        if(i===cur.length){ del=true; setTimeout(loop, pause); } else setTimeout(loop, speed);
      } else {
        typedEl.textContent = cur.slice(0, --i);
        if(i===0){ del=false; p=(p+1)%phrases.length; setTimeout(loop,200); } else setTimeout(loop, speed/1.2);
      }
    }
    setTimeout(loop,500);
    if(cursorEl) cursorEl.style.opacity = 1;
  })();
});
