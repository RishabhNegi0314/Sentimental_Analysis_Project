
function applyNeutralBlinkEffect(positiveCount, negativeCount) {
    const overallMood = document.getElementById('overall-mood');

    if (positiveCount === negativeCount && positiveCount > 0) {
        overallMood.textContent = 'Neutral';
        overallMood.style.color = '#ffffff';

        // Apply GSAP blinking animation
        gsap.to(overallMood, {
            opacity: 0.2,
            repeat: -1,
            yoyo: true,
            duration: 0.5,
            ease: "power1.inOut"
        });
    } else {
        // Stop animation if it's no longer neutral
        gsap.killTweensOf(overallMood);
        overallMood.style.opacity = 1;
    }
}

// Example usage: Call this function whenever sentiment counts change
// applyNeutralBlinkEffect(positiveCount, negativeCount);
