"""Prompts for synthetic fallacy-example generation. Edit AVOID as bad outputs show up."""

# Lessons learned from reviewing generated samples. Each line is injected into the prompt.
AVOID = [
    "Do not name the fallacy, or hint at it with words like 'fallacy', 'strawman', 'slippery slope', in the argument itself.",
    "Write only the text the person would say. No scaffolding labels such as 'Tweet:', 'Comment under a video:', 'Reply:', 'Op-ed:'.",
    "Do not use these overused names: Maya, Jamal, Jon, Priya, Ava, Ben, Tina, Alex, Jake, Mina, Rina, Tom, Jules. Prefer no names, or varied uncommon ones.",
    "Do not write a scripted textbook scene where one speaker states a universal claim, a second speaker announces a counterexample, and the first dismisses it. Real arguments are subtler: implied premises, single speaker, a reply that only partly quotes the other side.",
    "Avoid contrived universal claims like 'Every gamer on this team knows how to aim'. Use claims people actually make.",
    "Do not repeat the same sentence template across examples. Vary structure, length, and where the fallacy sits in the text.",
    "The fallacy must be visible in the written text alone. Never rely on tone of voice, emphasis, or context the reader cannot see.",
    "The speaker in the text must be the one COMMITTING the fallacy. Never write a speaker who is accusing someone else of it, complaining about it, or correcting it.",
    "Stay strictly inside the given definition. Do not drift to a neighbouring fallacy (e.g. 'it is old, so it is better' is Appeal to Tradition, not Appeal to Nature).",
    "Fallacies that respond to someone else's claim (straw man, red herring, tu quoque, ad hominem, ...) must show the other person's original claim in the text (a quote or dialogue turn) so the fallacy is visible, and the fallacious reply must be what the speaker says.",
    "Sincere tone. The speaker believes their argument. No jokes, sarcasm for effect, or absurd exaggeration beyond what real people post.",
    "Do not make the argument contain its own rebuttal or the correct reasoning.",
    "Across the examples, do not reuse the same connective pattern (for example 'then ... then ... eventually'). Chains, if used, differ in length and wording.",
    "Do not make the argument also contain a flat contradiction or factual impossibility that is not part of the fallacy.",
    "The explanation must start with exactly 'This is a <Name>:' or 'This is an <Name>:' using the given fallacy name, then explain in 2-3 sentences.",
]

GENERATE_SYSTEM = """You write training data for a model that spots logical fallacies in short social-media-style arguments.
You are given ONE fallacy (name, aliases, a plain definition, its general form). Write original examples of people committing it.

Requirements for every example:
- The argument is something a real person would plausibly post or say: 1-4 sentences, max ~80 words, natural voice, concrete topic.
- It clearly commits the target fallacy, and ideally no other fallacy more obviously.
- Follow the format assigned to each example number. The topic is an optional hint: ignore it whenever it makes the example less natural or drifts away from the definition (e.g. fallacies about random events need genuinely random situations like dice, lotteries, sports bets).
- Use your own words. Do not reuse well-known textbook examples.
- The explanation (2-3 sentences) says what the speaker did wrong, in plain language, referencing the specific argument.
  It starts with the fallacy name, e.g. "This is a Straw Man: ...".
{avoid}
Return JSON only."""

GENERATE_USER = """Fallacy: {name}
Also known as: {aka}
Definition: {description}
General form: {logical_form}
{guidance}
Write {n} different examples, one per assignment below:
{assignments}"""

JUDGE_SYSTEM = """You are a strict logic instructor reviewing a training example.
You are given a target fallacy and a candidate argument with an explanation. Decide:
- commits_target: does the argument clearly commit the target fallacy (not merely resemble it)?
- more_obvious_other_fallacy: name another fallacy ONLY if it is clearly a better label for this argument, else "".
- realistic: would a real person plausibly write this? (false for contrived, cartoonish, or strawmanned text)
- leaks_label: does the argument text itself name or hint at the fallacy?
- explanation_ok: is the explanation correct, specific to the argument, and starts with 'This is a/an <fallacy name>:'?
- speaker_commits: is the person whose words are shown the one committing the fallacy (false if they are accusing, quoting, or rebutting it)?
- templated: does it read like a scripted textbook scene or a contrived universal claim rather than something posted online?
- ambiguous: could a reasonable reader see it as NOT committing the fallacy (e.g. a fair inference, a joke, a plain preference)?
- issues: one short sentence describing the main problem, or "" if none.
First write `analysis`: 2-3 sentences stating who says what, what the fallacy would be, and whether the speaker commits it.
Then fill the booleans consistently with that analysis. Be harsh. Return JSON only."""

JUDGE_USER = """Target fallacy: {name}
Definition: {description}
{guidance}
Argument:
{argument}

Explanation:
{explanation}"""


# Per-fallacy guidance, shown to both generator and judge. Keyed by lowercase fallacy name.
# Add an entry only when generic rules keep producing drifted or invalid examples.
GUIDANCE = {
    "gambler's fallacy": "The process MUST be genuinely random and independent: coin flips, dice, roulette, lottery, slot machines, raffles, "
                         "card draws, coin-toss calls. Never skill, weather, human behaviour, prayer, or scheduling. "
                         "The speaker believes a streak makes the opposite outcome 'due' (or the streak 'cannot continue').",
    "appeal to nature": "The claim is that something is good, safe or better BECAUSE it is natural (plants, organic, unprocessed, 'chemical-free'), "
                        "or bad BECAUSE it is artificial/synthetic. Good topics: food, supplements, medicine, skincare, parenting, farming, pets. "
                        "Not about old-vs-new technology or tradition.",
    "fallacy of composition": "Moves from 'each part/member has property P' to 'the whole/group/team/system as a collective has P' "
                              "(e.g. every player is excellent, so the team must be excellent; each part is light, so the machine is light). "
                              "Not a generalisation from one or two samples, which is Hasty Generalization.",
    "equivocation": "One key word is used in two genuinely different dictionary senses across the argument (e.g. 'bank', 'light', 'right', 'critic'), "
                    "and the conclusion only follows if the senses are conflated. Both senses must be real and the argument must sound natural, "
                    "not a pun or wordplay joke.",
    'affirmative conclusion from a negative premise': "Write the syllogism EXPLICITLY as three short statements (two premises, then a conclusion) in everyday words, spoken by one person walking through their reasoning, e.g. 'No reptiles are warm-blooded. Some of my pets are reptiles. So some of my pets are warm-blooded.' Follow the formal pattern exactly. At least one premise is negative (No X are Y / Some X are not Y) and the conclusion is affirmative (All/Some ... are ...).",
    'exclusive premises': "Write the syllogism EXPLICITLY as three short statements (two premises, then a conclusion) in everyday words, spoken by one person walking through their reasoning, e.g. 'No reptiles are warm-blooded. Some of my pets are reptiles. So some of my pets are warm-blooded.' Follow the formal pattern exactly. BOTH premises are negative (No X are Y / Some X are not Y), and a conclusion is drawn anyway.",
    'illicit major': "Write the syllogism EXPLICITLY as three short statements (two premises, then a conclusion) in everyday words, spoken by one person walking through their reasoning, e.g. 'No reptiles are warm-blooded. Some of my pets are reptiles. So some of my pets are warm-blooded.' Follow the formal pattern exactly. Pattern: All A are B. No C are A. Therefore no C are B. (e.g. All dogs are mammals. No cats are dogs. So no cats are mammals.)",
    'fallacy of four terms': "Write the syllogism EXPLICITLY as three short statements (two premises, then a conclusion) in everyday words, spoken by one person walking through their reasoning, e.g. 'No reptiles are warm-blooded. Some of my pets are reptiles. So some of my pets are warm-blooded.' Follow the formal pattern exactly. The argument secretly has four terms instead of three: either two unrelated pairs of terms, or a middle term used in two different senses (e.g. 'All banks are near rivers. I keep my money in a bank. So my money is near a river.').",
    'negating antecedent and consequent': "Pattern: 'If P then Q. Therefore, if not P then not Q' (or the reverse). State the if-then premise and the invalid flipped conclusion in everyday words, e.g. 'If you study, you pass. So if you don't study, you won't pass.'",
    'gish gallop': 'ONE message firing 6-9 separate, mostly weak or unrelated claims in rapid succession (statistics, anecdotes, quotes, accusations), leaving no room to answer them all. Up to ~150 words is fine. Do not make any single claim detailed.',
    'homunculus fallacy': "The speaker 'explains' how a person/animal/system does X (sees, understands, decides, feels, speaks) by positing an inner agent, screen, or 'little person' that does X, which just moves the question one level down.",
    'stolen concept fallacy': 'The speaker tries to disprove or dismiss X while relying on X in the very argument (e.g. denying that words can have meaning, using words with meanings to say so; denying objective truth and presenting that as the truth).',
    'texas sharpshooter fallacy': "The speaker looks at data AFTER the fact, picks out a cluster or coincidence, and treats it as a meaningful pattern or prediction, ignoring the rest of the data (e.g. 'Five of the 40 winners were born in March, so March babies are lucky').",
    'alternative advance': "A sales or persuasion line that offers two options which are really the same outcome in different words (e.g. 'Would you like to start on Monday or on Tuesday?' when the person has not agreed to start). The speaker presents it as the only choice.",
    'deceptive sharing': "The speaker explicitly shares a vivid post/clip/story of a statistically RARE event (the frame is 'sharing this...') in order to make it seem common or typical right now. The share itself is the argument.",
    'ambiguity fallacy': "A whole PHRASE or sentence structure (not a single word) is ambiguous (e.g. 'flying planes can be dangerous', 'visiting relatives can be annoying', 'I saw her duck') and the argument only works by silently picking one reading.",
    'lying with statistics': 'The speaker quotes a number in a misleading way: relative vs absolute risk, tiny sample, missing baseline, cherry-picked time window, percentage of a small base, or an average that hides spread. The misleading use is visible from the text.',
    'subverted support': "The speaker offers a detailed explanation of WHY some phenomenon happens, but there is no evidence the phenomenon actually occurs (e.g. 'Why do cats always land on their feet when dropped from a window? because...' when the premise itself is unestablished).",
    'alphabet soup': 'A pile of 5+ unexplained acronyms/abbreviations/jargon packed into the text and used as the reason to trust the speaker or to overwhelm the listener, without actually supporting the claim.',
    'double standard': "The speaker themself applies two different standards to two comparable cases and doesn't notice (e.g. praises their own side for something and condemns the other side for the same thing). The speaker is NOT pointing out someone else's double standard.",
    'rationalization': 'The speaker gives a flattering or respectable reason for what they did or believe, while their own words let the reader see the real, less flattering motive (envy, fear, laziness, ego). The speaker is not aware of or admits only the false reason.',
    'argument by selective reading': 'Show a short message with several points (including one weak one), then a reply that responds ONLY to the weakest point as though it were the whole argument. The reply is the fallacious speaker.',
    'disjunction without a difference': "The speaker insists two options/labels/policies are different (or that one is acceptable and the other not) when they are the same in practice and differ only in wording (e.g. 'I'm not firing you, I'm letting you go').",
    'having your cake': 'The speaker answers in a deliberately noncommittal way that keeps both positions open and lets them later claim they were right whichever way things turn out.',
    'hypnotic bait and switch': "3-4 uncontroversial, true statements in a row (so the audience nods along), then a controversial claim stated as if it follows from them. The true statements are generic and don't actually support the claim.",
    'type token fallacy': "A word refers to a TYPE (a kind, e.g. 'the dodo', 'the Ford Model T') in one sentence and a TOKEN (one particular thing) in another, and the argument slides between them (e.g. 'The lion is endangered; there's a lion at our zoo; so our zoo lion is endangered').",
    'equivocation': "A single word is used with two clearly different dictionary meanings in two steps of the argument (e.g. 'light', 'bank', 'right', 'novel', 'party', 'law'). State the steps plainly, as a short chain 'X is Y (sense 1). Therefore Z (sense 2).' Don't use puns or wordplay.",
}
