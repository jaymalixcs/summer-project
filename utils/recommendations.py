"""
utils/recommendations.py
========================
Cluster interpretation text and marketing recommendation data.

Returns Python dicts/lists so Flask can render them in templates.
"""

# ─── Cluster ID → persona-content remap ──────────────────────────────────────
# IMPORTANT: K-Means does NOT guarantee that "cluster 0" will always be the
# high-income/low-spending group, or that "cluster 3" will always be the
# high-income/high-spending group. The actual numeric ID K-Means assigns to
# each group depends on the data and on how the algorithm's random starting
# points happen to land — it is essentially arbitrary.
#
# The dictionaries below (CLUSTER_PROFILES, RECOMMENDATIONS) describe FIXED
# personas ("Careful Spenders", "Target Customers", etc). This remap was
# built by actually running the pipeline, printing each cluster's real
# mean Annual_Income / Spending_Score (see utils/clustering.get_cluster_summary),
# and matching each one to the persona whose description fits those numbers.
# Without this step, the labels shown in the UI could easily describe the
# wrong group — e.g. calling a high-income/high-spending group "Budget
# Shoppers" just because it happened to get cluster_id 1.
CLUSTER_ID_REMAP = {
    0: 4,  # actual cluster 0 = medium income / medium spending  -> "Average Customers"
    1: 3,  # actual cluster 1 = high income   / high spending    -> "Target Customers"
    2: 2,  # actual cluster 2 = low income    / high spending    -> "Impulsive Buyers"
    3: 0,  # actual cluster 3 = high income   / low spending     -> "Careful Spenders"
    4: 1,  # actual cluster 4 = low income    / low spending     -> "Budget Shoppers"
}

# ─── Cluster profiles (names + personas) ─────────────────────────────────────
CLUSTER_PROFILES = {
    0: {
        "name"        : "Careful Spenders",
        "icon"        : "🏦",
        "color"       : "#4FC3F7",
        "income"      : "High",
        "spending"    : "Low",
        "age_group"   : "Middle-aged (35–55)",
        "persona"     : "Financially disciplined professionals who earn well but spend conservatively. They are value-conscious, research-driven buyers who prioritise necessity over luxury.",
        "behavior"    : [
            "Visits mall infrequently — usually for essential purchases",
            "Compares prices before making a decision",
            "Rarely impulsive; prefers planned purchases",
            "Responds well to long-term value propositions (warranties, quality guarantees)",
        ],
        "business_meaning": "This group represents untapped high-income potential. With the right incentives, they can be converted to higher-spending customers.",
    },
    1: {
        "name"        : "Budget Shoppers",
        "icon"        : "🛒",
        "color"       : "#FF8A65",
        "income"      : "Low",
        "spending"    : "Low",
        "age_group"   : "Older (40–70)",
        "persona"     : "Low-income customers who are careful about every rupee they spend. They shop only for necessities and are extremely price-sensitive.",
        "behavior"    : [
            "Shops mainly during sales and discount events",
            "Prefers budget brands and private labels",
            "Highly loyal to stores offering consistent value",
            "Word-of-mouth is a key driver for this group",
        ],
        "business_meaning": "A large, stable segment. While individual spend is low, volume and loyalty make them important for steady footfall.",
    },
    2: {
        "name"        : "Impulsive Buyers",
        "icon"        : "⚡",
        "color"       : "#CE93D8",
        "income"      : "Low",
        "spending"    : "High",
        "age_group"   : "Young (18–35)",
        "persona"     : "Young, trend-driven shoppers who spend more than their income level suggests. Lifestyle, social media influence, and peer pressure drive their purchases.",
        "behavior"    : [
            "Highly susceptible to trends and influencer marketing",
            "Frequent mall visitors with high basket sizes",
            "Prefers experiences over pure product value",
            "Emotionally driven purchase decisions",
        ],
        "business_meaning": "High risk of churn but also high potential for upselling. Engaging them through community and exclusive experiences builds loyalty.",
    },
    3: {
        "name"        : "Target Customers",
        "icon"        : "🎯",
        "color"       : "#81C784",
        "income"      : "High",
        "spending"    : "High",
        "age_group"   : "Young-to-mid (25–40)",
        "persona"     : "The ideal customer — high income and high spending. These are affluent professionals or business owners who enjoy premium experiences and products.",
        "behavior"    : [
            "Frequent, high-value shoppers",
            "Prefers premium brands and exclusive collections",
            "Loyal to quality — not particularly price-sensitive",
            "Responds well to personalisation and VIP treatment",
        ],
        "business_meaning": "The most valuable segment. Retaining them through loyalty programmes and personalised offers directly impacts revenue.",
    },
    4: {
        "name"        : "Average Customers",
        "icon"        : "📊",
        "color"       : "#FFD54F",
        "income"      : "Medium",
        "spending"    : "Medium",
        "age_group"   : "All ages",
        "persona"     : "The mainstream middle-market customer. Balanced income and spending behaviour, not extreme in either direction. The largest segment by count.",
        "behavior"    : [
            "Regular, moderate-frequency mall visitors",
            "Receptive to both value and premium messaging",
            "Tends to follow family or group decision-making",
            "Loyalty programmes with tangible rewards resonate well",
        ],
        "business_meaning": "A broad, stable base. Even small improvements in engagement or basket size across this group yield large aggregate revenue gains.",
    },
}


# ─── Marketing recommendations per cluster ───────────────────────────────────
RECOMMENDATIONS = {
    0: {  # Careful Spenders
        "strategies": [
            {
                "title"      : "Value Assurance Campaigns",
                "icon"       : "🔒",
                "description": "Emphasise quality, durability, and long-term ROI in your messaging. Offer extended warranties and money-back guarantees to reduce purchase hesitation.",
                "channels"   : ["Email newsletters", "In-store signage", "Loyalty app notifications"],
                "kpi"        : "Conversion rate on premium items",
            },
            {
                "title"      : "Exclusive Member Sales",
                "icon"       : "🎖️",
                "description": "Invite this segment to exclusive pre-sales or member-only events. Being treated as 'special' without heavy discounting aligns with their self-image.",
                "channels"   : ["Direct mail", "SMS invites", "Email"],
                "kpi"        : "Event attendance and in-event spend",
            },
            {
                "title"      : "Cross-sell Premium Upgrades",
                "icon"       : "⬆️",
                "description": "When they buy a mid-tier product, show a clear side-by-side comparison with a premium alternative. Highlight the long-term cost benefit.",
                "channels"   : ["POS displays", "Sales associate scripts", "E-commerce recommendations"],
                "kpi"        : "Upgrade conversion rate",
            },
        ],
        "summary": "Focus on building trust and demonstrating value. This segment needs a rational business case before they spend more.",
    },
    1: {  # Budget Shoppers
        "strategies": [
            {
                "title"      : "Loyalty Points & Cashback",
                "icon"       : "💰",
                "description": "Introduce a simple loyalty programme with cashback or points redeemable on next visit. Low-effort, high perceived value for budget-conscious shoppers.",
                "channels"   : ["Loyalty cards", "WhatsApp updates", "Receipt coupons"],
                "kpi"        : "Repeat visit frequency",
            },
            {
                "title"      : "Flash Sales & Limited-Time Offers",
                "icon"       : "⏰",
                "description": "Time-sensitive deals create urgency and drive visits. Price-sensitive customers respond strongly to '24-hour sale' or 'Today only' messaging.",
                "channels"   : ["Push notifications", "SMS", "Social media stories"],
                "kpi"        : "Footfall during sale periods",
            },
            {
                "title"      : "Bundle Deals",
                "icon"       : "📦",
                "description": "Offer product bundles that provide savings compared to individual purchase. This increases average basket size without appearing expensive.",
                "channels"   : ["In-store displays", "Weekly flyers"],
                "kpi"        : "Average transaction value",
            },
        ],
        "summary": "Price is the primary motivator. Loyalty and savings programmes keep this segment coming back consistently.",
    },
    2: {  # Impulsive Buyers
        "strategies": [
            {
                "title"      : "Social Media & Influencer Campaigns",
                "icon"       : "📱",
                "description": "Partner with micro-influencers and run Instagram/TikTok campaigns featuring new arrivals. This segment makes decisions based on social proof and trends.",
                "channels"   : ["Instagram", "TikTok", "YouTube Shorts"],
                "kpi"        : "Social engagement rate and referral traffic",
            },
            {
                "title"      : "BNPL & Easy Instalment Options",
                "icon"       : "💳",
                "description": "Offer Buy Now Pay Later or zero-interest instalments. This removes the income barrier for high-ticket items and enables this segment to spend more.",
                "channels"   : ["Checkout page", "In-store POS", "Mobile app"],
                "kpi"        : "High-value transaction count",
            },
            {
                "title"      : "Experience-Based Events",
                "icon"       : "🎉",
                "description": "Host in-store pop-ups, product launches, and experiential events. Young, trend-driven shoppers value experiences and create organic social content.",
                "channels"   : ["Event invites", "Social media", "Email"],
                "kpi"        : "Event attendance and social mentions",
            },
        ],
        "summary": "Lead with excitement, trends, and community. Remove financial friction with flexible payment options.",
    },
    3: {  # Target Customers
        "strategies": [
            {
                "title"      : "VIP Loyalty Programme",
                "icon"       : "👑",
                "description": "Create a tiered VIP programme with exclusive perks: priority access, concierge service, personal shopping assistants, and early-bird collections.",
                "channels"   : ["Private app", "Personal account managers", "Curated emails"],
                "kpi"        : "Programme retention rate and lifetime value",
            },
            {
                "title"      : "Personalised Recommendations",
                "icon"       : "✨",
                "description": "Use purchase history to send hyper-personalised product recommendations. Affluent shoppers appreciate curation over broad promotions.",
                "channels"   : ["Email", "In-app", "Personal shopper consultations"],
                "kpi"        : "Recommendation click-through and conversion rate",
            },
            {
                "title"      : "Premium Product Launches",
                "icon"       : "🚀",
                "description": "Invite this segment as first-access buyers for premium or limited-edition products. Exclusivity drives satisfaction and word-of-mouth in affluent circles.",
                "channels"   : ["Exclusive events", "Personal invitations", "Private previews"],
                "kpi"        : "First-day sales from VIP segment",
            },
        ],
        "summary": "Focus on exclusivity, personalisation, and premium experiences. ROI per customer is highest here — invest accordingly.",
    },
    4: {  # Average Customers
        "strategies": [
            {
                "title"      : "Standard Loyalty Programme",
                "icon"       : "⭐",
                "description": "Enrol this segment into a simple, transparent loyalty programme. Points, tiers, and birthday rewards keep engagement high across this broad group.",
                "channels"   : ["Email", "In-app", "Store loyalty cards"],
                "kpi"        : "Programme enrolment rate and active member count",
            },
            {
                "title"      : "Seasonal Promotions",
                "icon"       : "🎁",
                "description": "Target festive seasons, back-to-school, and major holidays with well-timed promotions. Average customers are receptive to both price and aspirational messaging.",
                "channels"   : ["Email", "SMS", "In-store"],
                "kpi"        : "Seasonal revenue uplift",
            },
            {
                "title"      : "Cross-Category Upselling",
                "icon"       : "🔄",
                "description": "Use purchase data to recommend complementary categories (e.g., clothing → accessories). Cross-category engagement increases basket size and visit diversity.",
                "channels"   : ["Email", "App notifications", "POS suggestions"],
                "kpi"        : "Cross-category purchase rate",
            },
        ],
        "summary": "Broad outreach works here. Consistency, relevance, and loyalty rewards keep this segment engaged and growing.",
    },
}


def get_cluster_profile(cluster_id: int) -> dict:
    """Return the profile dict for the given cluster ID."""
    return CLUSTER_PROFILES.get(CLUSTER_ID_REMAP.get(cluster_id, cluster_id), {})


def get_recommendations(cluster_id: int) -> dict:
    """Return the recommendations dict for the given cluster ID."""
    return RECOMMENDATIONS.get(CLUSTER_ID_REMAP.get(cluster_id, cluster_id), {})


def get_all_profiles() -> dict:
    """Return all cluster profiles, keyed by the ACTUAL K-Means cluster_id."""
    return {cid: CLUSTER_PROFILES[content_key] for cid, content_key in CLUSTER_ID_REMAP.items()}


def get_all_recommendations() -> dict:
    """Return all recommendations, keyed by the ACTUAL K-Means cluster_id."""
    return {cid: RECOMMENDATIONS[content_key] for cid, content_key in CLUSTER_ID_REMAP.items()}
