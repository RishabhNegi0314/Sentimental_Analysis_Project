
let tweets = [];
// API URL - change this to your actual API URL
const API_URL = 'http://localhost:5000/analyze';

// Slideshow variables
let positiveSlideshowIndex = 0;
let negativeSlideshowIndex = 0;

// Auto-slide variables
let carouselInterval;
let positiveSlideshowInterval;
let negativeSlideshowInterval;

// Function to update character count
function updateCharCount(textarea) {
    const maxLength = textarea.getAttribute('maxlength');
    const currentLength = textarea.value.length;
    const remaining = maxLength - currentLength;
    
    const charCountDiv = document.querySelector('.char-count');
    charCountDiv.textContent = `${remaining} characters remaining`;
    
    if (remaining < 50) {
        charCountDiv.style.color = '#f87171';
    } else {
        charCountDiv.style.color = '#8899a6';
    }
}

// Function to post a tweet
function postTweet() {
    const tweetInput = document.querySelector('.tweet-input');
    const tweetText = tweetInput.value.trim();
    
    if (tweetText) {
        tweets.push({
            text: tweetText,
            sentiment: null,
            confidence: null,
            author: "User_" + Math.floor(Math.random() * 1000)
        });
        tweetInput.value = '';
        updateCharCount(tweetInput);
    }
}

// Function to scroll the carousel
function scrollCarousel(direction) {
    const carousel = document.getElementById('tweets-carousel');
    const cardWidth = 280 + 15; // card width + gap
    carousel.scrollBy({
        left: direction * cardWidth,
        behavior: 'smooth'
    });
}

// Function to change slides
function changeSlide(containerId, controlsId, slideIndex, totalSlides) {
    const container = document.getElementById(containerId);
    const slides = container.getElementsByClassName('slideshow-item');
    const controls = document.getElementById(controlsId);
    
    // Hide all slides
    for (let i = 0; i < slides.length; i++) {
        slides[i].classList.remove('active');
    }
    
    // Update dots
    const dots = controls.getElementsByClassName('slideshow-dot');
    for (let i = 0; i < dots.length; i++) {
        dots[i].classList.remove('active');
    }
    
    // Show current slide and activate dot
    if (slides.length > 0) {
        slides[slideIndex].classList.add('active');
        dots[slideIndex].classList.add('active');
    }
    
    // Return index to use for updating global variables
    return slideIndex;
}

// Function to get confidence-based color
// Modified getConfidenceColor function
function getConfidenceColor(sentiment, confidence) {
    // If no sentiment or confidence, return transparent
    if (!sentiment || confidence === null) {
        return 'transparent';
    }
    
    // Calculate opacity based on confidence (minimum 0.3 for visibility)
    const opacity = Math.max(0.3, confidence);
    
    // For positive sentiment, return green with opacity
    if (sentiment === 'positive') {
        return `rgba(44, 242, 117, ${opacity})`;  // Bright green
    } 
    // For negative sentiment, return red with opacity
    else if (sentiment === 'negative') {
        return `rgba(248, 63, 63, ${opacity})`;   // Bright red
    }
    
    return 'transparent';
}

// Set up slideshow controls
function setupSlideshowControls(containerId, controlsId, slides) {
    const container = document.getElementById(containerId);
    const controls = document.getElementById(controlsId);
    controls.innerHTML = '';
    
    // Create dots for each slide
    for (let i = 0; i < slides.length; i++) {
        const dot = document.createElement('div');
        dot.className = 'slideshow-dot';
        if (i === 0) dot.classList.add('active');
        dot.addEventListener('click', function() {
            if (containerId === 'positive-slideshow') {
                positiveSlideshowIndex = i;
                changeSlide(containerId, controlsId, positiveSlideshowIndex, slides.length);
            } else {
                negativeSlideshowIndex = i;
                changeSlide(containerId, controlsId, negativeSlideshowIndex, slides.length);
            }
        });
        controls.appendChild(dot);
    }
}

// Function to start auto-sliding for carousel
function startCarouselAutoSlide() {
    // Clear any existing interval
    if (carouselInterval) {
        clearInterval(carouselInterval);
    }
    
    // Set new interval - changes slide every 5 seconds (5000ms)
    carouselInterval = setInterval(() => {
        scrollCarousel(1); // Scroll to the right
    }, 3000);
}

// Function to start auto-sliding for positive slideshow
function startPositiveSlideshowAutoSlide() {
    // Clear any existing interval
    if (positiveSlideshowInterval) {
        clearInterval(positiveSlideshowInterval);
    }
    
    // Set new interval - changes slide every 4 seconds
    positiveSlideshowInterval = setInterval(() => {
        const positiveSlides = document.getElementById('positive-slideshow').getElementsByClassName('slideshow-item');
        if (positiveSlides.length > 1) {
            positiveSlideshowIndex = (positiveSlideshowIndex + 1) % positiveSlides.length;
            changeSlide('positive-slideshow', 'positive-controls', positiveSlideshowIndex, positiveSlides.length);
        }
    }, 3000);
}

// Function to start auto-sliding for negative slideshow
function startNegativeSlideshowAutoSlide() {
    // Clear any existing interval
    if (negativeSlideshowInterval) {
        clearInterval(negativeSlideshowInterval);
    }
    
    // Set new interval - changes slide every 4 seconds
    negativeSlideshowInterval = setInterval(() => {
        const negativeSlides = document.getElementById('negative-slideshow').getElementsByClassName('slideshow-item');
        if (negativeSlides.length > 1) {
            negativeSlideshowIndex = (negativeSlideshowIndex + 1) % negativeSlides.length;
            changeSlide('negative-slideshow', 'negative-controls', negativeSlideshowIndex, negativeSlides.length);
        }
    }, 3000);
}

// Pause auto-sliding when user interacts with carousel
function setupPauseOnInteraction() {
    // Pause carousel when user hovers over it
    const carousel = document.getElementById('tweets-carousel');
    carousel.addEventListener('mouseenter', () => {
        if (carouselInterval) {
            clearInterval(carouselInterval);
        }
    });
    
    carousel.addEventListener('mouseleave', () => {
        startCarouselAutoSlide();
    });
    
    // Pause slideshows when user hovers over them
    const positiveSlideshow = document.getElementById('positive-slideshow');
    positiveSlideshow.addEventListener('mouseenter', () => {
        if (positiveSlideshowInterval) {
            clearInterval(positiveSlideshowInterval);
        }
    });
    
    positiveSlideshow.addEventListener('mouseleave', () => {
        startPositiveSlideshowAutoSlide();
    });
    
    const negativeSlideshow = document.getElementById('negative-slideshow');
    negativeSlideshow.addEventListener('mouseenter', () => {
        if (negativeSlideshowInterval) {
            clearInterval(negativeSlideshowInterval);
        }
    });
    
    negativeSlideshow.addEventListener('mouseleave', () => {
        startNegativeSlideshowAutoSlide();
    });
}

// works perfectly fine.
async function analyzeTweets() {
    if (tweets.length === 0) {
        alert('Please add some tweets first!');
        return;
    }

    const loader = document.querySelector('.loader');
    loader.classList.remove('hidden'); // Show loader

    const analyzeBtn = document.querySelector('.analyze-btn');
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';

    // Process each tweet that hasn't been analyzed yet
    for (let i = 0; i < tweets.length; i++) {
        if (tweets[i].sentiment === null) {
            // Call the API
            const result = await analyzeWithAPI(tweets[i].text);
            tweets[i].sentiment = result.sentiment;
            tweets[i].confidence = result.confidence;
        }
    }

    // Show the tweets section after analysis
    document.getElementById('tweets-section').style.display = 'block';

    // Show the analysis container
    document.getElementById('analysis-container').style.display = 'block';

    // Update displays
    updateTweetsCarousel();
    updateSlideshows();
    updateAnalysisDisplay();

    // Start auto-sliding for all components
    startCarouselAutoSlide();
    startPositiveSlideshowAutoSlide();
    startNegativeSlideshowAutoSlide();
    setupPauseOnInteraction();
    document.getElementById('tweets-section').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Tweet';

    loader.classList.add('hidden'); // Hide loader
}

// Update the tweets carousel
function updateTweetsCarousel() {
    const carouselContainer = document.getElementById('tweets-carousel');
    carouselContainer.innerHTML = '';
    
    tweets.forEach((tweet, index) => {
        const tweetElement = document.createElement('div');
        tweetElement.className = 'tweet';
        
        // Add hover effects with confidence-based backgrounds
        tweetElement.addEventListener('mouseenter', () => {
            const confidenceColor = getConfidenceColor(tweet.sentiment, tweet.confidence);
            tweetElement.style.backgroundColor = confidenceColor;
            tweetElement.style.transition = 'background-color 0.3s ease';
            
            // Adjust text color for readability based on background
            if (tweet.sentiment === 'positive' && tweet.confidence > 0.5) {
                tweetElement.style.color = '#1e1e1e'; // Dark text on light background
            } else if (tweet.sentiment === 'negative' && tweet.confidence > 0.5) {
                tweetElement.style.color = '#ffffff'; // Light text on dark background
            }
        });
        
        tweetElement.addEventListener('mouseleave', () => {
            tweetElement.style.backgroundColor = '';
            tweetElement.style.color = '';
        });
        
        let sentimentHtml = '';
        
        if (tweet.sentiment === 'loading') {
            sentimentHtml = '<span class="sentiment loading">[analyzing...]</span>';
        } else if (tweet.sentiment === 'error') {
            sentimentHtml = '<span class="sentiment error">[error]</span>';
        } else if (tweet.sentiment !== null) {
            sentimentHtml = `<span class="sentiment ${tweet.sentiment}">[${tweet.sentiment}]</span>`;
        }
        
        tweetElement.innerHTML = `
            <div class="tweet-header">
                <div class="avatar">
                    ${tweet.author.charAt(0)}
                </div>
                <div class="tweet-author">${tweet.author}</div>
            </div>
            <p>${tweet.text} ${sentimentHtml}</p>
            ${tweet.confidence ? `<div class="confidence-indicator">Confidence: ${Math.round(tweet.confidence * 100)}%</div>` : ''}
        `;
        
        carouselContainer.appendChild(tweetElement);
    });
}

// Update the slideshows
function updateSlideshows() {
    const positiveSlideshow = document.getElementById('positive-slideshow');
    const negativeSlideshow = document.getElementById('negative-slideshow');
    
    positiveSlideshow.innerHTML = '';
    negativeSlideshow.innerHTML = '';
    
    let positiveTweets = tweets.filter(tweet => tweet.sentiment === 'positive');
    let negativeTweets = tweets.filter(tweet => tweet.sentiment === 'negative');
    
    // Create positive slides
    if (positiveTweets.length === 0) {
        const slide = document.createElement('div');
        slide.className = 'slideshow-item active';
        slide.innerHTML = '<p class="no-tweets">No positive tweets to display</p>';
        positiveSlideshow.appendChild(slide);
    } else {
        positiveTweets.forEach((tweet, index) => {
            const slide = document.createElement('div');
            slide.className = 'slideshow-item';
            if (index === 0) slide.classList.add('active');
            
            slide.innerHTML = `
                <div class="tweet-header">
                    <div class="avatar">${tweet.author.charAt(0)}</div>
                    <div class="tweet-author">${tweet.author}</div>
                </div>
                <p>${tweet.text}</p>
                ${tweet.confidence ? `<div class="confidence-indicator">Confidence: ${Math.round(tweet.confidence * 100)}%</div>` : ''}
            `;
            
            positiveSlideshow.appendChild(slide);
        });
        
        // Set up slideshow controls
        setupSlideshowControls('positive-slideshow', 'positive-controls', positiveTweets);
    }
    
    // Create negative slides
    if (negativeTweets.length === 0) {
        const slide = document.createElement('div');
        slide.className = 'slideshow-item active';
        slide.innerHTML = '<p class="no-tweets">No negative tweets to display</p>';
        negativeSlideshow.appendChild(slide);
    } else {
        negativeTweets.forEach((tweet, index) => {
            const slide = document.createElement('div');
            slide.className = 'slideshow-item';
            if (index === 0) slide.classList.add('active');
            
            slide.innerHTML = `
                <div class="tweet-header">
                    <div class="avatar">${tweet.author.charAt(0)}</div>
                    <div class="tweet-author">${tweet.author}</div>
                </div>
                <p>${tweet.text}</p>
                ${tweet.confidence ? `<div class="confidence-indicator">Confidence: ${Math.round(tweet.confidence * 100)}%</div>` : ''}
            `;
            
            negativeSlideshow.appendChild(slide);
        });
        
        // Set up slideshow controls
        setupSlideshowControls('negative-slideshow', 'negative-controls', negativeTweets);
    }
}

// Update the analysis display
function updateAnalysisDisplay() {
    const overallMood = document.getElementById('overall-mood');
    const chartContainer = document.getElementById('chart-container');
    
    let positiveCount = tweets.filter(tweet => tweet.sentiment === 'positive').length;
    let negativeCount = tweets.filter(tweet => tweet.sentiment === 'negative').length;
    
    const total = positiveCount + negativeCount;

    // In your updateAnalysisDisplay function, add this:
    const moodbox = document.querySelector('.mood-box');
    moodbox.classList.remove('positive-mood', 'negative-mood', 'neutral-mood');

        if (positiveCount > negativeCount) {
            overallMood.textContent = 'Positive';
            overallMood.style.color = '#4ade80';
            moodbox.classList.add('positive-mood');
        } else if (negativeCount > positiveCount) {
            overallMood.textContent = 'Negative';
            overallMood.style.color = '#f87171';
            moodbox.classList.add('negative-mood');
        } else {
            overallMood.textContent = 'Neutral';
            overallMood.style.color = '#ffffff';
            moodbox.classList.add('neutral-mood');
        }

        // Remove existing canvas (if any)
        chartContainer.innerHTML = '<canvas id="sentimentChart"></canvas>';
        const ctx = document.getElementById('sentimentChart').getContext('2d');


        // Create a pie chart
        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Positive', 'Negative'],
                datasets: [{
                    data: [positiveCount, negativeCount],
                    backgroundColor: ['#2cf275c5', '#f83f3fd1'],
                    borderColor: '#1e1e1e',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { color: 'white' } // Set legend text color
                    }
                }
            }
        });
    }


// Function to analyze a single tweet using your API (keeping this unchanged)
async function analyzeWithAPI(tweetText) {
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: tweetText })
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const data = await response.json();
        return {
            sentiment: data.sentiment,
            confidence: data.confidence || 0.5  // Default confidence if not provided
        };
    } catch (error) {
        console.error('Error analyzing tweet:', error);
        return {
            sentiment: 'error',
            confidence: 0
        };
    }
}

const tweetInput = document.querySelector(".tweet-input");

tweetInput.addEventListener("input", () => {
    tweetInput.style.animation = "none"; // Stop animation when typing
});

tweetInput.addEventListener("blur", () => {
    tweetInput.style.animation = "typing 3.5s steps(40, end) infinite"; // Restart animation when not focused
});
