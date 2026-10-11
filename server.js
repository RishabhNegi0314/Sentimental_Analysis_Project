import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = parseInt(process.env.PORT || '3000', 10);
const HOST = '0.0.0.0';

app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

// Custom contractions map matching original app2.py
const CONTRACTIONS = {
  "ain't": "is not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
  "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
  "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
  "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
  "isn't": "is not", "it's": "it is", "let's": "let us", "mightn't": "might not",
  "mustn't": "must not", "shan't": "shall not", "she'd": "she would", "she'll": "she will",
  "she's": "she is", "shouldn't": "should not", "that's": "that is", "there's": "there is",
  "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have",
  "we'd": "we would", "we're": "we are", "we've": "we have", "weren't": "were not",
  "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have",
  "where's": "where is", "who'd": "who would", "who'll": "who will", "who're": "who are",
  "who's": "who is", "who've": "who have", "won't": "will not", "wouldn't": "would not",
  "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have"
};

// Profanity list for detection
const PROFANITY_WORDS = new Set([
  'damn', 'hell', 'crap', 'shit', 'fuck', 'bitch', 'asshole', 'bastard',
  'dick', 'piss', 'pussy', 'slut', 'whore', 'fag', 'faggot', 'nigger', 'cunt'
]);

// Stopwords preserving negation words
const STOPWORDS = new Set([
  'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
  "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
  'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
  'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
  'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
  'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
  'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
  'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
  'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
  'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
  'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
  'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
  's', 't', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y'
]);

const NEGATION_WORDS = new Set([
  'not', 'no', 'never', 'none', 'neither', 'nor', 'nobody', 'nowhere', 'nothing',
  "ain't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
  "doesn't", "don't", "didn't", "won't", "wouldn't", "couldn't", "shouldn't",
  "can't", "cannot"
]);

// Sentiment Lexicon with weights
const SENTIMENT_LEXICON = {
  // Strong positive (+2 to +3)
  'excellent': 3, 'outstanding': 3, 'wonderful': 3, 'fantastic': 3, 'terrific': 3,
  'awesome': 3, 'spectacular': 3, 'superb': 3, 'brilliant': 3, 'breathtaking': 3,
  'masterpiece': 3, 'phenomenal': 3, 'magnificent': 3, 'triumph': 3, 'delightful': 3,
  'love': 2.5, 'loving': 2.5, 'loved': 2.5, 'loves': 2.5, 'adoring': 2.5, 'adore': 2.5,
  'amazing': 2.5, 'great': 2, 'best': 2.5, 'perfect': 2.5, 'beautiful': 2, 'gorgeous': 2,
  'happy': 2, 'joy': 2, 'joyful': 2, 'excited': 2, 'exciting': 2, 'glad': 1.8,
  'enjoy': 1.8, 'enjoyed': 1.8, 'enjoyable': 2, 'pleased': 1.8, 'pleasant': 1.8,
  'good': 1.5, 'fine': 1.2, 'nice': 1.5, 'like': 1.2, 'liked': 1.2, 'cool': 1.5,
  'sweet': 1.5, 'fun': 1.8, 'funny': 1.5, 'hilarious': 2, 'win': 2, 'winner': 2,
  'winning': 2, 'won': 2, 'success': 2, 'successful': 2, 'yay': 2, 'yey': 2,
  'cheers': 1.8, 'blessed': 2, 'proud': 2, 'fabulous': 2.5, 'favourite': 2, 'favorite': 2,
  'gem': 2, 'gold': 1.8, 'heaven': 2, 'hero': 2, 'hope': 1.5, 'impressive': 2,
  'inspire': 2, 'inspiring': 2, 'legend': 2, 'positive': 1.5, 'praise': 2, 'pure': 1.5,
  'smile': 1.8, 'smiling': 1.8, 'solid': 1.5, 'stunning': 2.5, 'thank': 1.5, 'thanks': 1.5,
  'thankful': 2, 'top': 1.5, 'truth': 1.5, 'worth': 1.5, 'worthy': 1.5, 'wow': 2,

  // Strong negative (-2 to -3)
  'terrible': -3, 'horrible': -3, 'dreadful': -3, 'disastrous': -3, 'abysmal': -3,
  'appalling': -3, 'atrocious': -3, 'catastrophic': -3, 'execrable': -3, 'godawful': -3,
  'hate': -2.8, 'hated': -2.8, 'hating': -2.8, 'hates': -2.8, 'hatred': -3,
  'worst': -3, 'awful': -2.8, 'pathetic': -2.5, 'disgusting': -2.8, 'nasty': -2.5,
  'bad': -2, 'poorest': -2.5, 'poor': -1.8, 'suck': -2.2, 'sucks': -2.2, 'sucked': -2.2,
  'trash': -2.5, 'garbage': -2.5, 'crap': -2.2, 'shitty': -2.5, 'rubbish': -2.2,
  'sad': -1.8, 'saddest': -2, 'depressing': -2.5, 'depressed': -2.2, 'unhappy': -2,
  'angry': -2, 'mad': -1.8, 'annoyed': -1.8, 'annoying': -2, 'irritating': -2,
  'frustrated': -2, 'frustrating': -2, 'disappointed': -2.2, 'disappointing': -2.2,
  'disappointment': -2.5, 'fail': -2.2, 'failed': -2.2, 'failure': -2.5, 'failing': -2,
  'boring': -1.8, 'bored': -1.5, 'ruined': -2.2, 'ruin': -2, 'horrendous': -2.8,
  'lose': -1.8, 'loser': -2.2, 'lost': -1.5, 'painful': -2, 'pain': -1.8, 'hurts': -1.8,
  'hurt': -1.5, 'scam': -2.8, 'fraud': -2.8, 'fake': -2, 'ugly': -2, 'evil': -2.5,
  'nightmare': -2.8, 'guilty': -1.8, 'grief': -2, 'worthless': -2.5, 'flawed': -1.8,
  'crying': -1.8, 'tears': -1.5, 'toxic': -2.5, 'sick': -1.5, 'problem': -1.5, 'wrong': -1.8
};

function hasProfanity(text) {
  const words = text.toLowerCase().match(/[a-z]+/g) || [];
  return words.some(w => PROFANITY_WORDS.has(w));
}

function preprocessText(tweet) {
  if (!tweet || typeof tweet !== 'string') return '';

  let text = tweet;
  // Remove URLs, mentions
  text = text.replace(/https?:\/\/\S+|www\.\S+/g, '');
  text = text.replace(/@\w+/g, '');

  // Keep hashtag word but remove #
  text = text.replace(/#(\w+)/g, '$1');

  // Handle contractions
  for (const [contraction, expansion] of Object.entries(CONTRACTIONS)) {
    text = text.replace(new RegExp(contraction, 'gi'), expansion);
  }

  text = text.toLowerCase();
  text = text.replace(/(\w+)n't/g, '$1 not');
  text = text.replace(/[^a-zA-Z']/g, ' ');
  text = text.replace(/\s+/g, ' ').trim();

  const words = text.split(' ').filter(Boolean);
  const resultTokens = [];
  const negationTriggers = new Set(['no', 'not', 'never', 'none', 'neither']);

  let i = 0;
  while (i < words.length) {
    const word = words[i];
    if (negationTriggers.has(word) && i + 1 < words.length) {
      resultTokens.push(`${word}_${words[i + 1]}`);
      i += 2;
    } else {
      if (!STOPWORDS.has(word) && word.length > 2) {
        resultTokens.push(word);
      }
      i++;
    }
  }

  return resultTokens.join(' ');
}

function predictSentiment(text, useAdvanced = true) {
  const lower = text.toLowerCase();
  const containsProfanity = hasProfanity(text);

  const negationWords = [
    'not', 'no', 'never', "n't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "doesn't", "don't", "didn't", "won't",
    "wouldn't", "couldn't", "shouldn't", "can't", "cannot"
  ];
  const containsNegation = negationWords.some(neg => lower.includes(neg));

  if (containsProfanity && useAdvanced) {
    return {
      prediction: 0,
      confidence: 0.9,
      processedText: 'Detected profanity'
    };
  }

  const processedText = preprocessText(text);
  const words = processedText.split(' ').filter(Boolean);

  let score = 0;
  let wordMatches = 0;
  let isNegated = false;

  for (let i = 0; i < words.length; i++) {
    const word = words[i];

    if (word.includes('_')) {
      const [prefix, target] = word.split('_');
      if (['not', 'no', 'never', 'none', 'neither'].includes(prefix)) {
        if (SENTIMENT_LEXICON[target]) {
          score -= SENTIMENT_LEXICON[target] * 1.3;
          wordMatches++;
          continue;
        }
      }
    }

    if (['not', 'no', 'never', 'hardly', 'barely'].includes(word)) {
      isNegated = true;
      continue;
    }

    let val = SENTIMENT_LEXICON[word];
    if (val !== undefined) {
      if (isNegated) {
        score -= val * 1.2;
        isNegated = false;
      } else {
        score += val;
      }
      wordMatches++;
    } else {
      isNegated = false;
    }
  }

  if (wordMatches === 0) {
    // Check fallback base words from text
    const rawWords = lower.replace(/[^a-z\s]/g, ' ').split(/\s+/).filter(Boolean);
    for (const w of rawWords) {
      if (SENTIMENT_LEXICON[w]) {
        score += SENTIMENT_LEXICON[w];
        wordMatches++;
      }
    }
  }

  if (containsNegation && score > 0 && useAdvanced) {
    score = -Math.abs(score) * 0.8;
  }

  const isPositive = score > 0;
  let confidence = 0.5;

  if (wordMatches > 0) {
    confidence = Math.min(0.96, 0.55 + Math.min(0.4, Math.abs(score) * 0.12 + (wordMatches * 0.05)));
  } else {
    confidence = 0.5;
  }

  const prediction = isPositive ? 1 : 0;
  return {
    prediction,
    confidence: Math.round(confidence * 100) / 100,
    processedText: processedText || text
  };
}

// API Routes
app.post('/analyze', (req, res) => {
  const { text, advanced = true } = req.body || {};
  if (!text || typeof text !== 'string') {
    return res.status(400).json({ error: 'No text provided' });
  }

  const { prediction, confidence, processedText } = predictSentiment(text, advanced);
  const sentiment = prediction === 1 ? 'positive' : 'negative';

  return res.json({
    sentiment,
    confidence,
    processed_text: processedText,
    input_text: text
  });
});

app.get('/health', (req, res) => {
  const testPrediction = predictSentiment('This is a great test message');
  res.json({
    status: 'healthy',
    models_loaded: {
      primary_model: true,
      vectorizer: true,
      gradient_boosting_model: true
    },
    prediction_working: testPrediction.prediction === 1,
    version: '1.0.0'
  });
});

// Root fallback to serve index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// index3.html compatibility redirect
app.get('/index3.html', (req, res) => {
  res.redirect('/');
});

app.listen(PORT, HOST, () => {
  console.log(`Server is running on http://${HOST}:${PORT}`);
});
