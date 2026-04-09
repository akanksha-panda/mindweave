#server\emotions\emotion_data.py


# =========================
# . POSITIVE EMOTIONS (FULL FLAT LIST)
# =========================



POSITIVE = {

    # =========================
    #  LOVE / CONNECTION
    # =========================
    "love": [
        "affectionate","admiring","compassionate","caring","devoted",
        "empathetic","friendly","loving","openhearted","sympathetic",
        "tender","warm","loved","supported","touched"
    ],

    # =========================
    #  CONFIDENCE / POWER
    # =========================
    "confidence": [
        "adequate","assured","authentic","bold","brave","capable",
        "centered","certain","empowered","powerful","proud",
        "secure","strong","sure","reliable","tenacious","trusting"
    ],

    # =========================
    #  FOCUS / PRODUCTIVITY
    # =========================
    "focus": [
        "clear","focused","productive","responsible","efficient",
        "conscientious","aware","attentive","informed","listening"
    ],

    # =========================
    #  GROWTH / MOTIVATION
    # =========================
    "growth": [
        "hopeful","optimistic","passionate","perseverant",
        "motivated","stimulated","challenged","inspired"
    ],

    # =========================
    # . ENGAGEMENT / CURIOSITY
    # =========================
    "engagement": [
        "absorbed","alert","approachable","communicative",
        "connected","curious","eager","enthusiastic",
        "fascinated","flexible","interested","intrigued",
        "involved","responsive"
    ],

    # =========================
    #  ENERGY / VITALITY
    # =========================
    "energy": [
        "energetic","glowing","invigorated","lively","vibrant",
        "energized","rejuvenated","renewed","rested","restored",
        "revived","vigorous","dynamic"
    ],

    # =========================
    #  JOY / HAPPINESS
    # =========================
    "joy": [
        "amazed","animated","aroused","dazzled","ecstatic",
        "giddy","radiant","amused","cheerful","delighted",
        "elated","funny","glad","gleeful","happy",
        "jovial","jubilant","playful","pleased","spontaneous",
        "upbeat","uplifted","thrilled"
    ],

    # =========================
    #  PEACE / CALM
    # =========================
    "calm": [
        "at ease","calm","carefree","clearheaded","comfortable", "sorted"
        "content","easy","easygoing","free","fulfilled",
        "liberated","lighthearted","mellow","meditative",
        "observant","patient","present","quiet","relaxed",
        "relieved","satisfied","serene","still","tranquil"
    ],

    # =========================
    #  GRATITUDE
    # =========================
    "gratitude": [
        "appreciative","blessed","giving","grateful",
        "gratified","honored","privileged","thankful"
    ],

    # =========================
    #  CREATIVITY / REFLECTION
    # =========================
    "creativity": [
        "awed","blissful","creative","reflective"
    ],

    # =========================
    #  SOCIAL / KINDNESS
    # =========================
    "kindness": [
        "considerate","cooperative","gentle","helpful",
        "open minded","receptive","sincere","thoughtful",
        "tolerant","understanding"
    ],

    # =========================
    #  STABILITY / BALANCE
    # =========================
    "stability": [
        "stable","steady","grounded","harmonious"
    ],

    # =========================
    #  UNIQUE / EDGE CASES
    # =========================
    "misc_positive": [
        "excellent","genuine","glorious","keen","open",
        "self conscious","serious","sharing"
    ]
}


# =========================
# . NEGATIVE EMOTIONS (FULL IMAGE INCLUDED)
# =========================

# server/emotions/negative_emotions.py

NEGATIVE = {

    # =========================
    # AFRAID
    # =========================
    "afraid": [
        "afraid","apprehensive","dread","fearful","frightened","incapable",
        "insecure","mistrustful","panicked","paranoid","petrified","scared",
        "suspicious","terrified","wary","worried"
    ],

    # =========================
    # AVERSION
    # =========================
    "aversion": [
        "animosity","aggressive","attacked","bitter","close","controlling",
        "disgusted","dishonest","defensive","dislike","hate","horrified",
        "hostile","offended","repulsed","revengeful","stubborn","threatened","violent"
    ],

    # =========================
    # DISQUIET
    # =========================
    "disquiet": [
        "agitated","alarmed","avoiding","awkward","disconcerted","disturbed",
        "impatient","indecisive","negative","overwhelmed","perturbed",
        "pessimistic","preoccupied","restless","self critical",
        "self deprecating","shocked","surprised","troubled","uncomfortable",
        "uneasy","unnerved","unsettled","unsure","unworthy","upset"
    ],

    # =========================
    # FATIGUE
    # =========================
    "fatigue": [
        "beat","burnt out","crushed","depleted","dreadful","exhausted",
        "empty","fatigued","lethargic","listless","sleepy","tired","weak","worn out","unwell", "sick", "ill"
    ],

    # =========================
    # TENSE
    # =========================
    "tense": [
        "anxious","cranky","distressed","distraught","disturbed","fidgety",
        "frazzled","irritable","nervous","off","paralyzed","restless",
        "stressed","stressed out","tense","uptight"
    ],

    # =========================
    # ANNOYED
    # =========================
    "annoyed": [
        "aggravated","bummed out","burdened","complaining","displeased",
        "dissatisfied","exasperated","frustrated","grumpy","impatient",
        "irritated","moody"
    ],

    # =========================
    # CONFUSED
    # =========================
    "confused": [
        "ambivalent","baffled","dazed","disillusioned","doubtful","hesitant",
        "lost","mystified","perplexed","puzzled","stuck","torn"
    ],

    # =========================
    # ANGRY
    # =========================
    "angry": [
        "angry","enraged","furious","hateful","livid","outraged","mad",
        "mean","menaced","resentful","upset"
    ],

    # =========================
    # EMBARRASSED
    # =========================
    "embarrassed": [
        "ashamed","bullied","chagrined","embarrassed","guilty","humiliated",
        "insulted","intimidated","mortified","rejected"
    ],

    # =========================
    # PAIN
    # =========================
    "pain": [
        "agony","anguished","bullied","devastated","grief","heartbroken",
        "hurt","lonely","miserable","regretful","remorseful"
    ],

    # =========================
    # SAD
    # =========================
    "sad": [
        "alone","depressed","dejected","desperate","disappointed","discouraged",
        "discontented","disheartened","forlorn","gloomy","heavy hearted", "failure", "fail", 
        "heartbroken","hurt","lonely","miserable","morose","terrible"
    ],

    # =========================
    # VULNERABLE
    # =========================
    "vulnerable": [
        "fragile","guarded","helpless","hopeless","insecure","inferior",
        "leery","powerless","reserved","sensitive","shaky","uncertain",
        "useless","victimized","vulnerable"
    ],

    # =========================
    # DISCONNECTED
    # =========================
    "disconnected": [
        "alienated","aloof","apathetic","bored","cold","critical","detached",
        "distant","distracted","disconnected","indifferent","numb","removed",
        "resentful","rigid","resistant","sluggish","trapped","uninterested",
        "unpleasant","withdrawn"
    ],

    # =========================
    # YEARNING
    # =========================
    "yearning": [
        "envious","jealous","longing","nostalgic","pining","wistful"
    ]
}




EMOTION_PHRASES = {

    # =========================
    # SADNESS / LOW MOOD
    # =========================
    "sadness": [
        "a lump in my throat",
        "at a low ebb",
        "at my wit's end",
        "at the end of my rope",
        "beside myself",
        "blue monday",
        "broken-hearted",
        "close to tears",
        "crying my eyes out",
        "down and out",
        "down in the dumps",
        "feeling blue",
        "heavy-hearted",
        "in low spirits",
        "in tears",
        "not myself",
        "run down",
        "sick at heart",
        "thin-skinned",
        "under the weather",
        "worn out",

        # words
        "sad", "lonely", "heartbroken", "hopeless", "miserable",
        "regretful", "depressed", "discouraged", "disappointed",
        "overwhelmed", "jealous", "insecure", "embarrassed",
        "ashamed", "helpless", "isolated"
    ],

    # =========================
    # FATIGUE / LOW ENERGY
    # =========================
    "fatigue": [
        "dead tired",
        "dog-tired",
        "fed up",
        "running on fumes",
        "sick and tired",
        "worn out",
        "no energy",
        "drained",
        "exhausted",

        # words
        "tired", "bored", "pessimistic"
    ],

    # =========================
    # ANXIETY / STRESS
    # =========================
    "anxiety": [
        "break out in a cold sweat",
        "butterflies in my stomach",
        "heart in my mouth",
        "hold your breath",
        "in a cold sweat",
        "on edge",
        "on pins and needles",
        "lose sleep over it",
        "sweating bullets",
        "with bated breath",
        "worry myself sick",
        "lose sleep over it",

        # words
        "anxious", "worried", "nervous", "restless",
        "uneasy", "overwhelmed", "uncertain", "jumpy",
        "apprehensive", "paranoid"
    ],

    # =========================
    # FEAR
    # =========================
    "fear": [
        "frightened to death",
        "jump out of my skin",
        "make my blood run cold",
        "scared out of my wits",
        "scared stiff",
        "send shivers down my spine",
        "shaking like a leaf",
        "in a cold sweat",

        # words
        "scared", "fearful", "frightened", "startled", "timid"
    ],

    # =========================
    # ANGER
    # =========================
    "anger": [
        "fed up",
        "get on my nerves",
        "get worked up",
        "had it up to here",
        "lose my temper",
        "sick and tired",

        # words
        "angry", "annoyed", "irritated", "frustrated",
        "resentful", "impatient", "defensive",
        "hostile", "enraged", "irritable",
        "offended", "agitated", "furious",
        "bitter", "short-tempered", "argumentative",
        "moody", "vengeful", "defiant"
    ],

    # =========================
    # CONFUSION
    # =========================
    "confusion": [
        "caught between two stools",
        "not myself",
        "feel off",
        "i dont understand",
        "what should i do",

        # words
        "confused", "uncertain", "hesitant"
    ],

    # =========================
    # POSITIVE / HAPPY
    # =========================
    "positive": [
        "a weight off my shoulders",
        "all smiles",
        "as pleased as punch",
        "at peace",
        "breathe easy",
        "bright-eyed and bushy-tailed",
        "bursting with joy",
        "chin up",
        "cool as a cucumber",
        "every cloud has a silver lining",
        "full of beans",
        "grinning from ear to ear",
        "happy as a clam",
        "happy-go-lucky",
        "having a ball",
        "having the time of my life",
        "in high spirits",
        "in seventh heaven",
        "in stitches",
        "in the clear",
        "keep your cool",
        "light at the end of the tunnel",
        "look on the bright side",
        "on a high",
        "on cloud nine",
        "on the sunny side",
        "on top of the world",
        "out of the woods",
        "over the moon",
        "rest easy",
        "riding high",
        "right as rain",
        "rolling in the aisles",
        "sleep like a log",
        "take it easy",
        "tickled pink",
        "turn the corner",
        "walking on air",
        "whistling a happy tune",

        # words
        "happy", "joyful", "cheerful", "excited",
        "playful", "optimistic", "confident", "hopeful",
        "relaxed", "proud", "inspired", "loving",
        "grateful", "peaceful", "energetic", "friendly",
        "curious", "generous", "delighted", "thrilled",
        "festive", "giddy", "ecstatic"
    ],

    # =========================
    # LOVE / CONNECTION
    # =========================
    "love": [
        "affectionate", "romantic", "passionate", "caring",
        "attached", "compassionate", "flirty", "adoring",
        "devoted", "tender", "protective", "trusting",
        "admiring", "loyal"
    ],

    # =========================
    # CALM / PEACEFUL
    # =========================
    "calm": [
        "calm", "serene", "content", "balanced", "patient",
        "gentle", "tranquil", "mindful", "comfortable",
        "stable", "grounded", "soothed", "chilled", "harmonious"
    ]
}




# =========================
# . ROMANTIC PHRASES DATASET
# =========================

ROMANTIC_PHRASES = [
    "makes the heart feel full",
    "brings warmth to the soul",
    "feels like the best thing ever",
    "grows stronger with time",
    "becomes the reason behind every love song",
    "creates deep gratitude",
    "feels effortless and natural",
    "makes everything feel beautiful",
    "lights up the entire day",
    "carries stories within the eyes",
    "brings meaning into life",
    "becomes a missing piece",
    "creates emotional intimacy",
    "inspires deep connection",
    "draws the soul closer",
    "reflects vulnerability and honesty",
    "feels like a safe place",
    "brings calm and comfort",
    "fills life with joy",
    "feels like home",

    # attraction / fancy
    "makes the eyes shine",
    "creates a magnetic pull",
    "feels deeply captivating",
    "draws attention effortlessly",
    "holds emotional depth",
    "sparks curiosity",
    "feels enchanting",
    "carries quiet charm",
    "feels irresistible",
    "draws emotional focus",

    # admiration
    "inspires admiration",
    "reflects honesty and kindness",
    "shows strength through challenges",
    "carries inner beauty",
    "grows more admirable with time",
    "reflects authenticity",
    "shows emotional depth",
    "earns deep respect",
    "carries grace and patience",
    "inspires pride",

    # adoration
    "feels irreplaceable",
    "brings joy to everyday moments",
    "brightens every space",
    "becomes the favorite part of the day",
    "creates lasting happiness",
    "fills life with meaning",
    "becomes unforgettable",
    "feels deeply cherished",
    "holds emotional value",
    "brings constant comfort",

    # cherish
    "makes every moment feel special",
    "creates lasting memories",
    "brings emotional security",
    "strengthens connection daily",
    "holds deep personal value",
    "brings happiness consistently",
    "becomes emotionally grounding",
    "creates meaningful experiences",
    "feels worth holding onto",
    "strengthens emotional bonds",

    # deep romantic
    "makes the heart flutter",
    "creates emotional warmth",
    "inspires endless affection",
    "brings peace and comfort",
    "feels like destiny",
    "creates a sense of belonging",
    "brings emotional clarity",
    "strengthens inner happiness",
    "creates a deep bond",

    # general romantic signals
    "feels deeply connected",
    "creates emotional pull",
    "brings calm energy",
    "inspires affection",
    "creates attraction",
    "builds emotional closeness",
    "feels comforting",
    "creates attachment",
    "brings joy naturally",
    "inspires admiration",
    "feels grounding",
    "builds trust",
    "creates emotional safety",
    "strengthens connection",
    "feels meaningful"
]


# =========================
# 💎 ROMANTIC WORDS DATASET
# =========================

ROMANTIC_WORDS = [
    # love
    "affectionate", "devoted", "caring", "tender", "loving",
    "passionate", "attached", "adoring", "warm", "gentle",

    # attraction
    "captivating", "charming", "magnetic", "enchanting", "radiant",
    "mesmerizing", "irresistible", "alluring", "stunning", "graceful",

    # personality
    "genuine", "kind", "compassionate", "thoughtful", "supportive",
    "patient", "honest", "soft-hearted", "understanding", "nurturing",

    # deep emotional
    "irreplaceable", "meaningful", "profound", "intimate",
    "emotional", "heartfelt", "deep", "soulful", "sincere",

    # positive energy
    "calming", "uplifting", "inspiring", "energizing",
    "comforting", "stabilizing", "fulfilling",
    "grounding", "reassuring", "peaceful"
]


# =========================
# . OPTIONAL 
# =========================

ROMANTIC_DATA = {
    "r_phrases": ROMANTIC_PHRASES,
    "r_words": ROMANTIC_WORDS
}



# =========================
# . FLATTENED CATEGORY MAP
# =========================

FLAT_EMOTION_MAP = {}

# . POSITIVE
for category, words in POSITIVE.items():
    for word in words:
        FLAT_EMOTION_MAP[word] = {
            "category": category,
            "polarity": "positive"
        }

# . NEGATIVE
for category, words in NEGATIVE.items():
    for word in words:
        FLAT_EMOTION_MAP[word] = {
            "category": category,
            "polarity": "negative"
        }

# . PHRASES
for category, phrases in EMOTION_PHRASES.items():
    for phrase in phrases:
        FLAT_EMOTION_MAP[phrase] = {
            "category": category,
            "polarity": "positive" if category in ["positive", "love", "calm"] else "negative"
        }

# =========================
# . ROMANTIC → LOVE MAPPING
# =========================

LOVE_CONFIG = {
    "mood": +2,
    "energy": +1,
    "distortion": -1,
    "sentiment": +0.8
}

# . romantic phrases
for phrase in ROMANTIC_DATA["r_phrases"]:
    FLAT_EMOTION_MAP[phrase] = {
        "category": "love",
        "polarity": "positive",
        **LOVE_CONFIG
    }

# . romantic words
for word in ROMANTIC_DATA["r_words"]:
    FLAT_EMOTION_MAP[word] = {
        "category": "love",
        "polarity": "positive",
        **LOVE_CONFIG
    }

# =========================
# . ALL EMOTIONS 
# =========================

ALL_EMOTIONS = list(FLAT_EMOTION_MAP.keys())

ALL_EMOTIONS += ROMANTIC_DATA["r_phrases"]
ALL_EMOTIONS += ROMANTIC_DATA["r_words"]


# =========================
# . CATEGORY → FULL STATE (FINAL)
# =========================

CATEGORY_STATE_HINTS = {

    # =========================
    # . NEGATIVE
    # =========================

    "afraid":      {"mood": -2, "energy": -1, "distortion": +3, "sentiment": -0.7},
    "aversion":    {"mood": -2, "energy":  0, "distortion": +2, "sentiment": -0.6},
    "disquiet":    {"mood": -1, "energy": -1, "distortion": +3, "sentiment": -0.5},
    "fatigue":     {"mood": -1, "energy":  0, "distortion": +1, "sentiment": -0.4},
    "tense":       {"mood": -1, "energy": -1, "distortion": +4, "sentiment": -0.6},
    "annoyed":     {"mood": -1, "energy":  0, "distortion": +2, "sentiment": -0.5},
    "confused":    {"mood": -1, "energy":  0, "distortion": +3, "sentiment": -0.4},
    "angry":       {"mood": -2, "energy": +1, "distortion": +4, "sentiment": -0.7},
    "embarrassed": {"mood": -2, "energy": -1, "distortion": +1, "sentiment": -0.6},
    "pain":        {"mood": -3, "energy": -1, "distortion": +2, "sentiment": -0.9},
    "sad":         {"mood": -2, "energy": -1, "distortion": +1, "sentiment": -0.8},
    "vulnerable":  {"mood": -2, "energy": -1, "distortion": +2, "sentiment": -0.7},
    "disconnected":{"mood": -1, "energy":  0, "distortion": +1, "sentiment": -0.5},
    "yearning":    {"mood": -1, "energy":  0, "distortion": +1, "sentiment": -0.3},

    # =========================
    # . POSITIVE
    # =========================

    "love":        {"mood": +2, "energy": +1, "distortion": -1, "sentiment": +0.8},
    "confidence":  {"mood": +2, "energy": +1, "distortion": -1, "sentiment": +0.7},
    "focus":       {"mood": +1, "energy": +1, "distortion": -2, "sentiment": +0.5},
    "growth":      {"mood": +1, "energy": +1, "distortion": -1, "sentiment": +0.6},
    "engagement":  {"mood": +1, "energy": +1, "distortion": -1, "sentiment": +0.5},
    "energy":      {"mood": +1, "energy": +2, "distortion": -1, "sentiment": +0.6},
    "joy":         {"mood": +3, "energy": +2, "distortion": -2, "sentiment": +1.0},
    "calm":        {"mood": +2, "energy":  0, "distortion": -3, "sentiment": +0.7},
    "gratitude":   {"mood": +2, "energy": +1, "distortion": -1, "sentiment": +0.8},
    "creativity":  {"mood": +1, "energy": +1, "distortion": -1, "sentiment": +0.6},
    "kindness":    {"mood": +1, "energy":  0, "distortion": -1, "sentiment": +0.6},
    "stability":   {"mood": +1, "energy":  0, "distortion": -2, "sentiment": +0.5},
    "misc_positive":{"mood": +1, "energy": 0, "distortion":  0, "sentiment": +0.4},

    # =========================
    # . PHRASES
    # =========================

    "sadness":     {"mood": -2, "energy": -1, "distortion": +1, "sentiment": -0.8},
    "fatigue_phrase": {"mood": -1, "energy": 0, "distortion": +1, "sentiment": -0.5},
    "anxiety":     {"mood": -1, "energy": -1, "distortion": +4, "sentiment": -0.7},
    "fear":        {"mood": -2, "energy": -1, "distortion": +4, "sentiment": -0.8},
    "anger":       {"mood": -2, "energy": +1, "distortion": +4, "sentiment": -0.7},
    "confusion":   {"mood": -1, "energy":  0, "distortion": +3, "sentiment": -0.5},
    "positive":    {"mood": +2, "energy": +1, "distortion": -1, "sentiment": +0.7},
    "calm_phrase": {"mood": +2, "energy":  0, "distortion": -3, "sentiment": +0.7},
}