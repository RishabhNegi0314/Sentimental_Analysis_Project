// Global state
let tweets = [];
let bookmarks = [];
let searchHistory = [];
const SEARCH_HISTORY_KEY = 'twitter_sentiment_search_history';
const API_URL = '/analyze';

// Search History Management
function loadSearchHistory() {
    try {
        const stored = localStorage.getItem(SEARCH_HISTORY_KEY);
        if (stored) {
            const parsed = JSON.parse(stored);
            if (Array.isArray(parsed)) {
                searchHistory = parsed.slice(0, 10);
            }
        }
    } catch (e) {
        console.warn('Could not load search history from localStorage', e);
        searchHistory = [];
    }
    updateHistoryBadge();
    renderRecentHistoryChips();
}

function saveSearchQuery(queryText, sentiment = null, confidence = null) {
    if (!queryText || typeof queryText !== 'string') return;
    const cleanText = queryText.trim();
    if (!cleanText) return;

    // Check if query already exists (de-duplication)
    const existingIndex = searchHistory.findIndex(item => item.text.toLowerCase() === cleanText.toLowerCase());
    let itemTimestamp = new Date().toISOString();
    let preservedSentiment = sentiment;
    let preservedConfidence = confidence;

    if (existingIndex >= 0) {
        // Keep existing classification if current update has no classification
        if (!preservedSentiment && searchHistory[existingIndex].sentiment) {
            preservedSentiment = searchHistory[existingIndex].sentiment;
            preservedConfidence = searchHistory[existingIndex].confidence;
        }
        searchHistory.splice(existingIndex, 1);
    }

    const newItem = {
        id: 'hist_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
        text: cleanText,
        sentiment: preservedSentiment,
        confidence: preservedConfidence,
        timestamp: itemTimestamp
    };

    // Prepend to top and restrict to 10
    searchHistory.unshift(newItem);
    searchHistory = searchHistory.slice(0, 10);

    try {
        localStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(searchHistory));
    } catch (e) {
        console.warn('Failed to save search history', e);
    }

    updateHistoryBadge();
    renderRecentHistoryChips();
    
    const historyModal = document.getElementById('historyModal');
    if (historyModal && historyModal.classList.contains('open')) {
        renderHistoryList();
    }
}

function updateHistoryBadge() {
    const badge = document.getElementById('historyCount');
    if (badge) {
        badge.textContent = searchHistory.length;
    }
}

function renderRecentHistoryChips() {
    const bar = document.getElementById('recentHistoryBar');
    const container = document.getElementById('recentHistoryChips');
    if (!bar || !container) return;

    if (searchHistory.length === 0) {
        bar.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    bar.style.display = 'flex';
    // Show top 3 recent queries
    const recent = searchHistory.slice(0, 3);
    container.innerHTML = recent.map(item => `
        <button type="button" class="history-chip" onclick="reanalyzeHistoryItem('${item.id}')" title="Re-analyze: ${escapeHtml(item.text)}">
            <span class="history-chip-text">${escapeHtml(item.text)}</span>
            <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="opacity: 0.7;">
                <polyline points="23 4 23 10 17 10"></polyline>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
            </svg>
        </button>
    `).join('');
}

function openHistoryModal() {
    renderHistoryList();
    const modal = document.getElementById('historyModal');
    if (modal) modal.classList.add('open');
}

function renderHistoryList() {
    const listEl = document.getElementById('historyList');
    const clearBtn = document.getElementById('historyClearBtn');
    if (!listEl) return;

    if (searchHistory.length === 0) {
        if (clearBtn) clearBtn.style.display = 'none';
        listEl.innerHTML = `
            <div style="text-align: center; padding: 36px 0; color: var(--text-dim);">
                <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 10px; opacity: 0.6;">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                <p style="font-weight: 600; color: var(--text-main); font-size: 14px;">No Search History</p>
                <p style="font-size: 12px; margin-top: 4px;">Queries you post and analyze will automatically be stored here.</p>
            </div>
        `;
        return;
    }

    if (clearBtn) clearBtn.style.display = 'inline-block';

    listEl.innerHTML = searchHistory.map((item, idx) => {
        const timeAgo = formatTimeAgo(item.timestamp);
        const sentimentClass = item.sentiment ? item.sentiment : 'neutral';
        const sentimentText = item.sentiment ? item.sentiment.toUpperCase() : 'PENDING';
        const confidenceText = item.confidence ? `· ${Math.round(item.confidence * 100)}%` : '';

        return `
            <div class="history-item">
                <div class="history-item-content">
                    <div class="history-item-query">${escapeHtml(item.text)}</div>
                    <div class="history-item-meta">
                        <span class="sentiment-badge ${sentimentClass}" style="padding: 2px 8px; font-size: 10px;">${sentimentText} ${confidenceText}</span>
                        <span>${timeAgo}</span>
                    </div>
                </div>
                <div class="history-item-actions">
                    <button type="button" class="history-reanalyze-btn" onclick="reanalyzeHistoryItem('${item.id}')" title="Re-analyze query">
                        <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="5 3 19 12 5 21 5 3"></polygon>
                        </svg>
                        Re-analyze
                    </button>
                    <button type="button" class="history-delete-btn" onclick="deleteHistoryItem('${item.id}')" title="Remove from history" aria-label="Delete">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

function reanalyzeHistoryItem(id) {
    const item = searchHistory.find(h => h.id === id);
    if (!item) return;

    closeModal('historyModal');

    // Populate into textarea
    const tweetInput = document.querySelector('.tweet-input');
    if (tweetInput) {
        tweetInput.value = item.text;
        updateCharCount(tweetInput);
    }

    // Add to active tweets
    const newTweet = {
        id: 'tweet_' + Date.now() + '_' + Math.floor(Math.random() * 1000),
        text: item.text,
        sentiment: null,
        confidence: null,
        author: "User_" + Math.floor(Math.random() * 900 + 100),
        bookmarked: false,
        timestamp: new Date()
    };
    tweets.push(newTweet);

    // Bump item to front of history
    saveSearchQuery(item.text, item.sentiment, item.confidence);

    const snippet = item.text.length > 32 ? item.text.slice(0, 32) + '...' : item.text;
    showNotification(`Re-analyzing: "${snippet}"`);
    analyzeTweets();
}

function deleteHistoryItem(id) {
    searchHistory = searchHistory.filter(h => h.id !== id);
    try {
        localStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(searchHistory));
    } catch (e) {
        console.warn('Failed to save search history', e);
    }
    updateHistoryBadge();
    renderRecentHistoryChips();
    renderHistoryList();
    showNotification('Removed item from search history');
}

function clearSearchHistory() {
    searchHistory = [];
    try {
        localStorage.removeItem(SEARCH_HISTORY_KEY);
    } catch (e) {
        console.warn('Failed to clear search history', e);
    }
    updateHistoryBadge();
    renderRecentHistoryChips();
    renderHistoryList();
    showNotification('Search history cleared');
}

function formatTimeAgo(dateString) {
    if (!dateString) return 'recently';
    const date = new Date(dateString);
    const now = new Date();
    const diffSeconds = Math.floor((now - date) / 1000);

    if (diffSeconds < 60) return 'Just now';
    const diffMinutes = Math.floor(diffSeconds / 60);
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

// Slideshow variables
let positiveSlideshowIndex = 0;
let negativeSlideshowIndex = 0;

// Auto-slide variables
let carouselInterval;
let positiveSlideshowInterval;
let negativeSlideshowInterval;

// Toast notification helper
function showNotification(msg) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.remove('hidden');
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3200);
}

// Function to update character count
function updateCharCount(textarea) {
    const maxLength = parseInt(textarea.getAttribute('maxlength') || '280', 10);
    const currentLength = textarea.value.length;
    const remaining = maxLength - currentLength;
    
    const charCountDiv = document.querySelector('.char-count');
    if (!charCountDiv) return;
    charCountDiv.textContent = `${remaining} characters remaining`;
    
    if (remaining < 30) {
        charCountDiv.style.color = '#f87171';
    } else {
        charCountDiv.style.color = '#94a3b8';
    }
}

// Quick Sample Prompts
function loadSampleTweet(type) {
    const tweetInput = document.querySelector('.tweet-input');
    if (!tweetInput) return;
    
    const samples = {
        positive: [
            "Just launched our new product version and the initial user feedback is absolutely phenomenal! Loving the positive vibe! 🚀",
            "This was by far the best customer service experience I have ever had. Super fast, friendly and helpful!",
            "Such a beautiful sunny day, feeling extremely grateful and happy today!"
        ],
        negative: [
            "Flight got delayed 5 hours without explanation. Worst customer support imaginable, absolutely terrible.",
            "App keeps crashing every two minutes after the new update. Terrible UX, extremely frustrated and disappointed.",
            "Terrible product quality, broke after two days. Do not buy this garbage."
        ],
        mixed: [
            "The movie had stunning visuals and great acting, but the pacing felt really slow in the second half.",
            "Great atmosphere and delicious food at this cafe, but unfortunately the service was quite disorganized.",
            "Not bad at all for the price, though there is still plenty of room for improvement."
        ]
    };

    const choices = samples[type] || samples.positive;
    const picked = choices[Math.floor(Math.random() * choices.length)];
    tweetInput.value = picked;
    updateCharCount(tweetInput);
    tweetInput.focus();
}

// Load a full sample thread to test Trend Analysis timeline
function loadSampleThread() {
    const thread = [
        { text: "Just announced our major platform update! Absolutely ecstatic to share this with everyone, loving the positive energy! 🚀", timeOffset: 90 },
        { text: "Encountered a few login glitches on mobile right after release, quite frustrating for early testers.", timeOffset: 70 },
        { text: "Hotfix deployed in record time! Incredible speed and dedicated engineering team, outstanding work!", timeOffset: 45 },
        { text: "Customer sentiment is rebounding strongly, users are praising the new glassmorphic UI design and smooth charts!", timeOffset: 25 },
        { text: "Still need to refine tablet portrait view, but overall this is easily our most enjoyable update to date.", timeOffset: 5 }
    ];

    const now = Date.now();
    thread.forEach((item, idx) => {
        tweets.push({
            id: 'tweet_' + (now - idx * 1000) + '_' + Math.floor(Math.random() * 1000),
            text: item.text,
            sentiment: null,
            confidence: null,
            author: "User_" + Math.floor(Math.random() * 900 + 100),
            bookmarked: false,
            timestamp: new Date(now - item.timeOffset * 60 * 1000)
        });
    });

    showNotification('Loaded 5 chronological tweets! Click "Analyze Sentiment" to plot the Trend line.');
}

// Function to post a tweet
function postTweet() {
    const tweetInput = document.querySelector('.tweet-input');
    const tweetText = tweetInput.value.trim();
    
    if (tweetText) {
        const newTweet = {
            id: 'tweet_' + Date.now() + '_' + Math.floor(Math.random() * 1000),
            text: tweetText,
            sentiment: null,
            confidence: null,
            author: "User_" + Math.floor(Math.random() * 900 + 100),
            bookmarked: false,
            timestamp: new Date()
        };
        tweets.push(newTweet);
        saveSearchQuery(tweetText, null, null);
        tweetInput.value = '';
        updateCharCount(tweetInput);
        showNotification('Tweet added to list! Click "Analyze Sentiment" to evaluate.');
    } else {
        showNotification('Please type something in the tweet box first.');
    }
}

// Function to scroll the carousel
function scrollCarousel(direction) {
    const carousel = document.getElementById('tweets-carousel');
    if (!carousel) return;
    const cardWidth = 300 + 16; // card width + gap
    carousel.scrollBy({
        left: direction * cardWidth,
        behavior: 'smooth'
    });
}

// Function to change slides
function changeSlide(containerId, controlsId, slideIndex, totalSlides) {
    const container = document.getElementById(containerId);
    if (!container) return slideIndex;
    const slides = container.getElementsByClassName('slideshow-item');
    const controls = document.getElementById(controlsId);
    
    // Hide all slides
    for (let i = 0; i < slides.length; i++) {
        slides[i].classList.remove('active');
    }
    
    // Update dots
    if (controls) {
        const dots = controls.getElementsByClassName('slideshow-dot');
        for (let i = 0; i < dots.length; i++) {
            dots[i].classList.remove('active');
        }
        if (dots[slideIndex]) {
            dots[slideIndex].classList.add('active');
        }
    }
    
    // Show current slide
    if (slides.length > 0 && slides[slideIndex]) {
        slides[slideIndex].classList.add('active');
    }
    
    return slideIndex;
}

// Set up slideshow controls
function setupSlideshowControls(containerId, controlsId, slides) {
    const controls = document.getElementById(controlsId);
    if (!controls) return;
    controls.innerHTML = '';
    
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

// Auto-slide functions
function startCarouselAutoSlide() {
    if (carouselInterval) clearInterval(carouselInterval);
    carouselInterval = setInterval(() => {
        scrollCarousel(1);
    }, 4500);
}

function startPositiveSlideshowAutoSlide() {
    if (positiveSlideshowInterval) clearInterval(positiveSlideshowInterval);
    positiveSlideshowInterval = setInterval(() => {
        const positiveSlides = document.getElementById('positive-slideshow')?.getElementsByClassName('slideshow-item');
        if (positiveSlides && positiveSlides.length > 1) {
            positiveSlideshowIndex = (positiveSlideshowIndex + 1) % positiveSlides.length;
            changeSlide('positive-slideshow', 'positive-controls', positiveSlideshowIndex, positiveSlides.length);
        }
    }, 3800);
}

function startNegativeSlideshowAutoSlide() {
    if (negativeSlideshowInterval) clearInterval(negativeSlideshowInterval);
    negativeSlideshowInterval = setInterval(() => {
        const negativeSlides = document.getElementById('negative-slideshow')?.getElementsByClassName('slideshow-item');
        if (negativeSlides && negativeSlides.length > 1) {
            negativeSlideshowIndex = (negativeSlideshowIndex + 1) % negativeSlides.length;
            changeSlide('negative-slideshow', 'negative-controls', negativeSlideshowIndex, negativeSlides.length);
        }
    }, 3800);
}

function setupPauseOnInteraction() {
    const carousel = document.getElementById('tweets-carousel');
    if (carousel) {
        carousel.addEventListener('mouseenter', () => {
            if (carouselInterval) clearInterval(carouselInterval);
        });
        carousel.addEventListener('mouseleave', () => {
            startCarouselAutoSlide();
        });
    }
    
    const positiveSlideshow = document.getElementById('positive-slideshow');
    if (positiveSlideshow) {
        positiveSlideshow.addEventListener('mouseenter', () => {
            if (positiveSlideshowInterval) clearInterval(positiveSlideshowInterval);
        });
        positiveSlideshow.addEventListener('mouseleave', () => {
            startPositiveSlideshowAutoSlide();
        });
    }
    
    const negativeSlideshow = document.getElementById('negative-slideshow');
    if (negativeSlideshow) {
        negativeSlideshow.addEventListener('mouseenter', () => {
            if (negativeSlideshowInterval) clearInterval(negativeSlideshowInterval);
        });
        negativeSlideshow.addEventListener('mouseleave', () => {
            startNegativeSlideshowAutoSlide();
        });
    }
}

// Analyze tweets
async function analyzeTweets() {
    const tweetInput = document.querySelector('.tweet-input');
    // If user has text in input box but hasn't clicked "Add Tweet", post it first
    if (tweetInput && tweetInput.value.trim().length > 0) {
        postTweet();
    }

    if (tweets.length === 0) {
        showNotification('Please enter and add a tweet first!');
        return;
    }

    const loader = document.querySelector('.loader');
    if (loader) loader.classList.remove('hidden');

    const analyzeBtn = document.querySelector('.analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
    }

    // Process each tweet that hasn't been analyzed yet
    for (let i = 0; i < tweets.length; i++) {
        if (tweets[i].sentiment === null) {
            const result = await analyzeWithAPI(tweets[i].text);
            tweets[i].sentiment = result.sentiment;
            tweets[i].confidence = result.confidence;
            // Update classified sentiment in search history!
            saveSearchQuery(tweets[i].text, result.sentiment, result.confidence);
        }
    }

    // Show the tweets section
    const tweetsSection = document.getElementById('tweets-section');
    if (tweetsSection) tweetsSection.style.display = 'block';

    // Show the analysis container
    const analysisContainer = document.getElementById('analysis-container');
    if (analysisContainer) analysisContainer.style.display = 'block';

    // Update displays
    updateTweetsCarousel();
    updateSlideshows();
    updateAnalysisDisplay();

    // Start auto-sliding
    startCarouselAutoSlide();
    startPositiveSlideshowAutoSlide();
    startNegativeSlideshowAutoSlide();
    setupPauseOnInteraction();

    if (tweetsSection) {
        tweetsSection.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }

    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            Analyze Sentiment
        `;
    }

    if (loader) loader.classList.add('hidden');
}

// Update the tweets carousel with Glassmorphism cards
function updateTweetsCarousel() {
    const carouselContainer = document.getElementById('tweets-carousel');
    if (!carouselContainer) return;
    carouselContainer.innerHTML = '';
    
    tweets.forEach((tweet) => {
        const tweetElement = document.createElement('div');
        const sentimentClass = tweet.sentiment === 'positive' ? 'tweet-positive' : 
                               tweet.sentiment === 'negative' ? 'tweet-negative' : '';
        tweetElement.className = `tweet ${sentimentClass}`;
        
        const isBookmarked = bookmarks.some(b => b.id === tweet.id);
        const sentimentBadge = tweet.sentiment ? 
            `<span class="sentiment-badge ${tweet.sentiment}">${tweet.sentiment}</span>` : 
            `<span class="sentiment-badge neutral">Pending</span>`;
        
        const confidenceDisplay = tweet.confidence ? 
            `<span class="confidence-indicator">${Math.round(tweet.confidence * 100)}% match</span>` : '';

        tweetElement.innerHTML = `
            <div class="tweet-card-top">
                <div class="tweet-header">
                    <div class="avatar">${tweet.author.charAt(0)}</div>
                    <div class="tweet-author">${tweet.author}</div>
                </div>
                <div class="card-actions">
                    <button class="card-bookmark-btn ${isBookmarked ? 'active' : ''}" onclick="toggleBookmark('${tweet.id}')" title="Bookmark Tweet" aria-label="Bookmark">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="${isBookmarked ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path>
                        </svg>
                    </button>
                </div>
            </div>
            <p class="tweet-text-content">${escapeHtml(tweet.text)}</p>
            <div class="tweet-card-bottom">
                ${sentimentBadge}
                ${confidenceDisplay}
            </div>
        `;
        
        carouselContainer.appendChild(tweetElement);
    });
}

// Update the slideshows
function updateSlideshows() {
    const positiveSlideshow = document.getElementById('positive-slideshow');
    const negativeSlideshow = document.getElementById('negative-slideshow');
    if (!positiveSlideshow || !negativeSlideshow) return;
    
    positiveSlideshow.innerHTML = '';
    negativeSlideshow.innerHTML = '';
    
    const positiveTweets = tweets.filter(tweet => tweet.sentiment === 'positive');
    const negativeTweets = tweets.filter(tweet => tweet.sentiment === 'negative');
    
    // Positive slides
    if (positiveTweets.length === 0) {
        positiveSlideshow.innerHTML = `
            <div class="slideshow-item active">
                <p class="no-tweets">No positive tweets recorded yet</p>
            </div>
        `;
    } else {
        positiveTweets.forEach((tweet, index) => {
            const slide = document.createElement('div');
            slide.className = `slideshow-item ${index === 0 ? 'active' : ''}`;
            slide.innerHTML = `
                <div class="tweet-header">
                    <div class="avatar" style="border-color: rgba(44, 242, 117, 0.4);">${tweet.author.charAt(0)}</div>
                    <div class="tweet-author">${tweet.author}</div>
                </div>
                <p>${escapeHtml(tweet.text)}</p>
                <div class="confidence-indicator" style="color: #4ade80;">Confidence: ${Math.round((tweet.confidence || 0.8) * 100)}%</div>
            `;
            positiveSlideshow.appendChild(slide);
        });
        setupSlideshowControls('positive-slideshow', 'positive-controls', positiveTweets);
    }
    
    // Negative slides
    if (negativeTweets.length === 0) {
        negativeSlideshow.innerHTML = `
            <div class="slideshow-item active">
                <p class="no-tweets">No negative tweets recorded yet</p>
            </div>
        `;
    } else {
        negativeTweets.forEach((tweet, index) => {
            const slide = document.createElement('div');
            slide.className = `slideshow-item ${index === 0 ? 'active' : ''}`;
            slide.innerHTML = `
                <div class="tweet-header">
                    <div class="avatar" style="border-color: rgba(248, 63, 63, 0.4);">${tweet.author.charAt(0)}</div>
                    <div class="tweet-author">${tweet.author}</div>
                </div>
                <p>${escapeHtml(tweet.text)}</p>
                <div class="confidence-indicator" style="color: #f87171;">Confidence: ${Math.round((tweet.confidence || 0.8) * 100)}%</div>
            `;
            negativeSlideshow.appendChild(slide);
        });
        setupSlideshowControls('negative-slideshow', 'negative-controls', negativeTweets);
    }
}

// Update the analysis display and Trend Analysis Line Chart
let chartInstance = null;
let currentChartView = 'trend';

function switchChartView(view) {
    currentChartView = view;
    const tabTrend = document.getElementById('tabTrend');
    const tabDistribution = document.getElementById('tabDistribution');
    if (tabTrend && tabDistribution) {
        if (view === 'trend') {
            tabTrend.classList.add('active');
            tabDistribution.classList.remove('active');
        } else {
            tabTrend.classList.remove('active');
            tabDistribution.classList.add('active');
        }
    }
    renderActiveChart();
}

function formatTweetTime(timestamp, index, total) {
    if (timestamp instanceof Date && !isNaN(timestamp)) {
        return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    const d = new Date(Date.now() - (total - 1 - index) * 12 * 60 * 1000);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function updateAnalysisDisplay() {
    const overallMood = document.getElementById('overall-mood');
    const moodbox = document.querySelector('.mood-box');
    if (!overallMood || !moodbox) return;
    
    const analyzedTweets = tweets.filter(tweet => tweet.sentiment !== null);
    const positiveCount = analyzedTweets.filter(tweet => tweet.sentiment === 'positive').length;
    const negativeCount = analyzedTweets.filter(tweet => tweet.sentiment === 'negative').length;
    
    moodbox.classList.remove('positive-mood', 'negative-mood', 'neutral-mood');

    if (positiveCount > negativeCount) {
        overallMood.textContent = 'Predominantly Positive';
        overallMood.style.color = '#4ade80';
        moodbox.classList.add('positive-mood');
    } else if (negativeCount > positiveCount) {
        overallMood.textContent = 'Predominantly Negative';
        overallMood.style.color = '#f87171';
        moodbox.classList.add('negative-mood');
    } else {
        overallMood.textContent = 'Balanced / Neutral';
        overallMood.style.color = '#38bdf8';
        moodbox.classList.add('neutral-mood');
    }

    // Update Telemetry Grid
    updateTelemetry(analyzedTweets);

    // Render the chart (Line Trend or Ratio)
    renderActiveChart();
}

function updateTelemetry(analyzedTweets) {
    const telemetryNetPolarity = document.getElementById('telemetryNetPolarity');
    const telemetryMomentum = document.getElementById('telemetryMomentum');
    const telemetryPeakConfidence = document.getElementById('telemetryPeakConfidence');
    const telemetryDataPoints = document.getElementById('telemetryDataPoints');
    const trendDirectionBadge = document.getElementById('trendDirectionBadge');
    const trendDirectionText = document.getElementById('trendDirectionText');

    if (analyzedTweets.length === 0) return;

    // Calculate individual polarities and cumulative net score
    let cumulativeScore = 0;
    let peakConfidence = 0;
    let peakSentiment = 'positive';

    analyzedTweets.forEach((t) => {
        const conf = t.confidence || 0.6;
        const pol = t.sentiment === 'positive' ? conf : -conf;
        cumulativeScore += pol;
        if (conf > peakConfidence) {
            peakConfidence = conf;
            peakSentiment = t.sentiment;
        }
    });

    const netScore = cumulativeScore / analyzedTweets.length;
    const sign = netScore > 0 ? '+' : '';
    const formattedNet = `${sign}${netScore.toFixed(2)}`;

    if (telemetryNetPolarity) telemetryNetPolarity.textContent = formattedNet;
    if (telemetryPeakConfidence) {
        telemetryPeakConfidence.textContent = `${Math.round(peakConfidence * 100)}% (${peakSentiment})`;
        telemetryPeakConfidence.style.color = peakSentiment === 'positive' ? '#4ade80' : '#f87171';
    }
    if (telemetryDataPoints) telemetryDataPoints.textContent = `${analyzedTweets.length} tweets`;

    // Trend Direction & Momentum
    if (trendDirectionBadge && trendDirectionText && telemetryMomentum) {
        trendDirectionBadge.classList.remove('positive', 'negative', 'neutral');
        if (netScore > 0.15) {
            trendDirectionBadge.classList.add('positive');
            trendDirectionText.textContent = `↗ Upward Polarity (${formattedNet})`;
            telemetryMomentum.textContent = `Positive (${formattedNet})`;
            telemetryMomentum.style.color = '#4ade80';
        } else if (netScore < -0.15) {
            trendDirectionBadge.classList.add('negative');
            trendDirectionText.textContent = `↘ Downward Polarity (${formattedNet})`;
            telemetryMomentum.textContent = `Negative (${formattedNet})`;
            telemetryMomentum.style.color = '#f87171';
        } else {
            trendDirectionBadge.classList.add('neutral');
            trendDirectionText.textContent = `→ Stable Polarity (${formattedNet})`;
            telemetryMomentum.textContent = `Neutral (${formattedNet})`;
            telemetryMomentum.style.color = '#38bdf8';
        }
    }
}

function renderActiveChart() {
    const canvas = document.getElementById('sentimentChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    if (chartInstance) {
        chartInstance.destroy();
    }

    const analyzedTweets = tweets.filter(tweet => tweet.sentiment !== null);
    if (analyzedTweets.length === 0) return;

    if (currentChartView === 'trend') {
        // --- TREND ANALYSIS LINE CHART ---
        // Calculate polarity and cumulative trajectory over time
        const labels = [];
        const trajectoryData = [];
        const pointColors = [];
        const individualPolarityData = [];

        let runningSum = 0;
        analyzedTweets.forEach((t, index) => {
            const timeStr = formatTweetTime(t.timestamp, index, analyzedTweets.length);
            labels.push(`Tweet #${index + 1} (${timeStr})`);

            const conf = t.confidence || 0.6;
            const pol = t.sentiment === 'positive' ? conf : -conf;
            individualPolarityData.push(pol);

            runningSum += pol;
            const avgPol = runningSum / (index + 1);
            trajectoryData.push(parseFloat(avgPol.toFixed(2)));

            pointColors.push(t.sentiment === 'positive' ? '#2cf275' : '#f83f3f');
        });

        // Gradient fill for line chart
        const gradient = ctx.createLinearGradient(0, 0, 0, 260);
        gradient.addColorStop(0, 'rgba(56, 189, 248, 0.28)');
        gradient.addColorStop(0.5, 'rgba(56, 189, 248, 0.08)');
        gradient.addColorStop(1, 'rgba(248, 63, 63, 0.05)');

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Net Sentiment Polarity Trajectory',
                        data: trajectoryData,
                        borderColor: '#38bdf8',
                        borderWidth: 3,
                        backgroundColor: gradient,
                        fill: true,
                        tension: 0.35,
                        pointRadius: 6,
                        pointHoverRadius: 9,
                        pointBackgroundColor: pointColors,
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2
                    },
                    {
                        label: 'Individual Tweet Polarity',
                        data: individualPolarityData,
                        borderColor: 'rgba(255, 255, 255, 0.25)',
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        pointRadius: 4,
                        pointHoverRadius: 7,
                        pointBackgroundColor: pointColors,
                        pointBorderColor: 'transparent',
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    y: {
                        min: -1.0,
                        max: 1.0,
                        grid: {
                            color: (context) => context.tick.value === 0 ? 'rgba(255, 255, 255, 0.3)' : 'rgba(255, 255, 255, 0.07)',
                            lineWidth: (context) => context.tick.value === 0 ? 1.5 : 1
                        },
                        ticks: {
                            color: '#94a3b8',
                            font: { size: 11, weight: '600' },
                            callback: (value) => {
                                if (value === 1) return '+1.0 (Positive)';
                                if (value === 0) return ' 0.0 (Neutral)';
                                if (value === -1) return '-1.0 (Negative)';
                                return (value > 0 ? '+' : '') + value.toFixed(1);
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.04)'
                        },
                        ticks: {
                            color: '#cbd5e1',
                            font: { size: 11, weight: '500' },
                            maxRotation: 25,
                            minRotation: 0
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        align: 'end',
                        labels: {
                            color: '#cbd5e1',
                            font: { size: 11, weight: '600' },
                            usePointStyle: true,
                            pointStyle: 'circle',
                            padding: 12
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleColor: '#ffffff',
                        bodyColor: '#cbd5e1',
                        borderColor: 'rgba(255, 255, 255, 0.18)',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 12,
                        callbacks: {
                            title: (items) => {
                                const idx = items[0].dataIndex;
                                const t = analyzedTweets[idx];
                                return `Tweet #${idx + 1} • ${t.author}`;
                            },
                            label: (context) => {
                                const idx = context.dataIndex;
                                const t = analyzedTweets[idx];
                                if (context.datasetIndex === 0) {
                                    const sign = context.parsed.y > 0 ? '+' : '';
                                    return `Running Trend: ${sign}${context.parsed.y.toFixed(2)}`;
                                } else {
                                    const confPct = Math.round((t.confidence || 0.7) * 100);
                                    return `Tweet Sentiment: ${t.sentiment.toUpperCase()} (${confPct}%)`;
                                }
                            },
                            afterBody: (items) => {
                                const idx = items[0].dataIndex;
                                const t = analyzedTweets[idx];
                                const snippet = t.text.length > 60 ? t.text.slice(0, 60) + '...' : t.text;
                                return [`"${snippet}"`];
                            }
                        }
                    }
                }
            }
        });
    } else {
        // --- DISTRIBUTION RATIO DOUGHNUT CHART ---
        const positiveCount = analyzedTweets.filter(tweet => tweet.sentiment === 'positive').length;
        const negativeCount = analyzedTweets.filter(tweet => tweet.sentiment === 'negative').length;

        chartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive Sentiment', 'Negative Sentiment'],
                datasets: [{
                    data: [positiveCount || 1, negativeCount || 0],
                    backgroundColor: [
                        'rgba(44, 242, 117, 0.85)',
                        'rgba(248, 63, 63, 0.85)'
                    ],
                    borderColor: [
                        'rgba(44, 242, 117, 1)',
                        'rgba(248, 63, 63, 1)'
                    ],
                    borderWidth: 1.5,
                    hoverOffset: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '62%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#cbd5e1',
                            font: { size: 12, weight: '600' },
                            padding: 16,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleColor: '#ffffff',
                        bodyColor: '#cbd5e1',
                        borderColor: 'rgba(255, 255, 255, 0.18)',
                        borderWidth: 1,
                        padding: 10,
                        cornerRadius: 10
                    }
                }
            }
        });
    }
}

// Function to analyze a single tweet via backend API
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
            confidence: data.confidence || 0.6
        };
    } catch (error) {
        console.error('Error analyzing tweet:', error);
        return {
            sentiment: 'positive',
            confidence: 0.5
        };
    }
}

// Bookmarking System
function toggleBookmark(tweetId) {
    const tweet = tweets.find(t => t.id === tweetId);
    if (!tweet) return;

    const existingIndex = bookmarks.findIndex(b => b.id === tweetId);
    if (existingIndex >= 0) {
        bookmarks.splice(existingIndex, 1);
        showNotification('Removed from bookmarks');
    } else {
        bookmarks.push({ ...tweet });
        showNotification('Saved to bookmarks!');
    }

    updateBookmarkCount();
    updateTweetsCarousel();
    renderBookmarksList();
}

function updateBookmarkCount() {
    const countEl = document.getElementById('bookmarkCount');
    if (countEl) {
        countEl.textContent = bookmarks.length;
    }
}

function openBookmarksModal() {
    renderBookmarksList();
    const modal = document.getElementById('bookmarksModal');
    if (modal) modal.classList.add('open');
}

function renderBookmarksList() {
    const listEl = document.getElementById('bookmarksList');
    if (!listEl) return;

    if (bookmarks.length === 0) {
        listEl.innerHTML = `
            <p style="color: var(--text-dim); text-align: center; padding: 25px 0;">
                No saved tweets yet. Click the bookmark ribbon icon on any tweet card to store it here.
            </p>
        `;
        return;
    }

    listEl.innerHTML = bookmarks.map(b => `
        <div class="bookmark-item">
            <div style="flex: 1;">
                <div class="bookmark-text">${escapeHtml(b.text)}</div>
                <div class="bookmark-meta">
                    <span class="sentiment-badge ${b.sentiment || 'neutral'}">${b.sentiment || 'unclassified'}</span>
                    <span style="font-size: 11px; color: var(--text-dim);">${b.author}</span>
                </div>
            </div>
            <button class="bookmark-delete-btn" onclick="toggleBookmark('${b.id}')" title="Delete">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="3 6 5 6 21 6"></polyline>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                </svg>
            </button>
        </div>
    `).join('');
}

// Session Analytics Modal
function openProfileModal() {
    const statTotal = document.getElementById('statTotal');
    const statPositive = document.getElementById('statPositive');
    const statNegative = document.getElementById('statNegative');
    const statConfidence = document.getElementById('statConfidence');

    const total = tweets.length;
    const positive = tweets.filter(t => t.sentiment === 'positive').length;
    const negative = tweets.filter(t => t.sentiment === 'negative').length;
    
    let avgConf = 0;
    const analyzed = tweets.filter(t => t.confidence !== null);
    if (analyzed.length > 0) {
        avgConf = Math.round((analyzed.reduce((sum, t) => sum + (t.confidence || 0), 0) / analyzed.length) * 100);
    }

    if (statTotal) statTotal.textContent = total;
    if (statPositive) statPositive.textContent = positive;
    if (statNegative) statNegative.textContent = negative;
    if (statConfidence) statConfidence.textContent = `${avgConf}%`;

    const modal = document.getElementById('profileModal');
    if (modal) modal.classList.add('open');
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.remove('open');
}

function closeModalOnBackdrop(event, modalId) {
    if (event.target.id === modalId) {
        closeModal(modalId);
    }
}

// Helper utility to escape HTML
function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Export Analyzed Session Results as JSON or TXT
function exportResults(format = 'json') {
    if (tweets.length === 0) {
        showNotification('No tweets available to export. Add and analyze tweets first!');
        return;
    }

    const analyzedTweets = tweets.filter(t => t.sentiment !== null);
    if (analyzedTweets.length === 0) {
        showNotification('Please analyze your tweets first before exporting.');
        return;
    }

    const positiveCount = analyzedTweets.filter(t => t.sentiment === 'positive').length;
    const negativeCount = analyzedTweets.filter(t => t.sentiment === 'negative').length;
    const totalCount = analyzedTweets.length;
    const positivePercent = totalCount > 0 ? ((positiveCount / totalCount) * 100).toFixed(1) + '%' : '0%';
    const negativePercent = totalCount > 0 ? ((negativeCount / totalCount) * 100).toFixed(1) + '%' : '0%';
    const avgConfidence = totalCount > 0 
        ? ((analyzedTweets.reduce((acc, t) => acc + (t.confidence || 0), 0) / totalCount) * 100).toFixed(1) + '%'
        : '0%';
    
    let overallSentiment = 'Neutral / Balanced';
    if (positiveCount > negativeCount) overallSentiment = 'Predominantly Positive';
    else if (negativeCount > positiveCount) overallSentiment = 'Predominantly Negative';

    const now = new Date();
    const timestamp = now.toISOString();
    const dateStr = now.toISOString().slice(0, 10);

    if (format === 'json') {
        const exportData = {
            project: "Twitter Sentimental Analysis",
            exportedAt: timestamp,
            summary: {
                totalAnalyzedTweets: totalCount,
                positiveTweets: positiveCount,
                negativeTweets: negativeCount,
                positiveRatio: positivePercent,
                negativeRatio: negativePercent,
                averageConfidence: avgConfidence,
                overallSentiment: overallSentiment
            },
            tweets: analyzedTweets.map(t => ({
                id: t.id,
                author: t.author,
                text: t.text,
                sentiment: t.sentiment,
                confidence: t.confidence ? Math.round(t.confidence * 100) / 100 : null,
                confidencePercentage: t.confidence ? `${Math.round(t.confidence * 100)}%` : null,
                bookmarked: bookmarks.some(b => b.id === t.id)
            }))
        };

        const jsonString = JSON.stringify(exportData, null, 2);
        downloadFile(jsonString, `twitter-sentiment-analysis-${dateStr}.json`, 'application/json');
        showNotification('Exported session results as JSON file!');
    } else {
        // Formatted text report
        let report = `================================================================================\n`;
        report += `                 TWITTER SENTIMENTAL ANALYSIS SESSION REPORT\n`;
        report += `================================================================================\n`;
        report += `Generated: ${now.toUTCString()}\n\n`;
        report += `-------------------------------- SESSION SUMMARY --------------------------------\n`;
        report += `Total Analyzed Tweets : ${totalCount}\n`;
        report += `Overall Sentiment     : ${overallSentiment}\n`;
        report += `Positive Tweets       : ${positiveCount} (${positivePercent})\n`;
        report += `Negative Tweets       : ${negativeCount} (${negativePercent})\n`;
        report += `Average Confidence    : ${avgConfidence}\n\n`;
        report += `------------------------- DETAILED TWEET CLASSIFICATIONS ------------------------\n\n`;

        analyzedTweets.forEach((t, idx) => {
            const conf = t.confidence ? `${Math.round(t.confidence * 100)}%` : 'N/A';
            const isBookmarked = bookmarks.some(b => b.id === t.id) ? 'Yes' : 'No';
            report += `[#${idx + 1}] Author     : ${t.author}\n`;
            report += `     Sentiment  : ${t.sentiment.toUpperCase()} (Confidence: ${conf})\n`;
            report += `     Bookmarked : ${isBookmarked}\n`;
            report += `     Tweet Text : "${t.text.replace(/\r?\n/g, ' ')}"\n`;
            report += `--------------------------------------------------------------------------------\n`;
        });

        report += `\nEnd of Analysis Report.\n`;
        downloadFile(report, `twitter-sentiment-analysis-${dateStr}.txt`, 'text/plain;charset=utf-8');
        showNotification('Exported session report as Text file!');
    }
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 150);
}

// Expose modal and helper functions globally for inline onclick handlers
window.loadSampleTweet = loadSampleTweet;
window.postTweet = postTweet;
window.analyzeTweets = analyzeTweets;
window.scrollCarousel = scrollCarousel;
window.toggleBookmark = toggleBookmark;
window.openBookmarksModal = openBookmarksModal;
window.openProfileModal = openProfileModal;
window.closeModal = closeModal;
window.closeModalOnBackdrop = closeModalOnBackdrop;
window.updateCharCount = updateCharCount;
window.exportResults = exportResults;
window.switchChartView = switchChartView;
window.loadSampleThread = loadSampleThread;
window.openHistoryModal = openHistoryModal;
window.reanalyzeHistoryItem = reanalyzeHistoryItem;
window.deleteHistoryItem = deleteHistoryItem;
window.clearSearchHistory = clearSearchHistory;

// Initialize on startup
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadSearchHistory);
} else {
    loadSearchHistory();
}
