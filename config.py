"""
RAG Agent Configuration
=======================
Central configuration file for all agent settings.
All values are configurable and can be overridden via environment variables or API parameters.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# PRESET PROFILES
# =============================================================================

class AgentPreset(Enum):
    """Pre-configured agent profiles for different use cases"""
    QUICK = "quick"
    BALANCED = "balanced"
    DEEP_RESEARCH = "deep_research"
    CREATIVE = "creative"
    CUSTOM = "custom"


PRESET_CONFIGS = {
    "quick": {
        "name": "⚡ Quick Answer",
        "description": "Fast response, focused sources",
        "temperature": 0.3,
        "max_iterations": 6,
        "top_k": 6,
        "rerank_top_n": 5,
        "max_conversation_history": 10,
    },
    "balanced": {
        "name": "⭐ Balanced",
        "description": "Good depth, accurate & natural (Recommended)",
        "temperature": 0.3,
        "max_iterations": 12,
        "top_k": 12,
        "rerank_top_n": 8,
        "max_conversation_history": 15,
    },
    "deep_research": {
        "name": "🔬 Deep Research",
        "description": "Thorough analysis, all sources (slower)",
        "temperature": 0.2,
        "max_iterations": 20,
        "top_k": 18,
        "rerank_top_n": 10,
        "max_conversation_history": 20,
    },
    "creative": {
        "name": "🎨 Creative",
        "description": "More varied responses, brainstorming",
        "temperature": 0.7,
        "max_iterations": 10,
        "top_k": 10,
        "rerank_top_n": 6,
        "max_conversation_history": 12,
    },
}


# =============================================================================
# DEFAULT CONFIGURATION VALUES
# =============================================================================

@dataclass
class AgentConfig:
    """Main configuration for the RAG Agent"""
    
    # --- Agent Behavior ---
    temperature: float = 0.3
    temperature_min: float = 0.0
    temperature_max: float = 1.0
    
    max_iterations: int = 12
    max_iterations_min: int = 3
    max_iterations_max: int = 25
    
    # --- Context Management ---
    max_conversation_history: int = 15
    max_content_per_chunk: int = 4000
    max_content_per_message: int = 3000
    max_tool_result_size: int = 30000
    
    # --- Search Configuration ---
    top_k: int = 12
    top_k_min: int = 3
    top_k_max: int = 25
    
    rerank_top_n: int = 8
    rerank_top_n_min: int = 3
    rerank_top_n_max: int = 15
    
    # --- Memory Settings ---
    memory_relevance_threshold: float = 0.75
    include_memories_in_context: bool = True
    max_memories_in_context: int = 5
    
    # --- Chunking Settings ---
    chunk_size: int = 3500
    chunk_overlap: int = 350
    semantic_threshold: int = 75
    table_chunk_size: int = 6000  # Larger for tables to keep them whole
    combine_text_under_n_chars: int = 300
    
    # --- Cost & Safety Controls ---
    max_tokens_per_request: int = 16000
    request_timeout: int = 90
    rate_limit_per_user: int = 100
    
    # --- Playwright Settings ---
    playwright_wait_time: int = 5000
    playwright_scroll_steps: int = 8
    playwright_extra_wait: int = 2000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "max_conversation_history": self.max_conversation_history,
            "max_content_per_chunk": self.max_content_per_chunk,
            "top_k": self.top_k,
            "rerank_top_n": self.rerank_top_n,
            "memory_relevance_threshold": self.memory_relevance_threshold,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
    
    @classmethod
    def from_preset(cls, preset: str) -> 'AgentConfig':
        """Create config from a preset name"""
        if preset not in PRESET_CONFIGS:
            preset = "balanced"
        
        preset_values = PRESET_CONFIGS[preset]
        return cls(
            temperature=preset_values.get("temperature", 0.3),
            max_iterations=preset_values.get("max_iterations", 12),
            top_k=preset_values.get("top_k", 12),
            rerank_top_n=preset_values.get("rerank_top_n", 8),
            max_conversation_history=preset_values.get("max_conversation_history", 15),
        )
    
    @classmethod
    def from_user_settings(cls, settings: Dict[str, Any]) -> 'AgentConfig':
        """Create config from user-provided settings"""
        config = cls()
        
        # Apply user settings with validation
        if "temperature" in settings:
            config.temperature = max(config.temperature_min, 
                                    min(config.temperature_max, float(settings["temperature"])))
        
        if "max_iterations" in settings:
            config.max_iterations = max(config.max_iterations_min,
                                       min(config.max_iterations_max, int(settings["max_iterations"])))
        
        if "top_k" in settings:
            config.top_k = max(config.top_k_min,
                              min(config.top_k_max, int(settings["top_k"])))
        
        if "rerank_top_n" in settings:
            config.rerank_top_n = max(config.rerank_top_n_min,
                                     min(config.rerank_top_n_max, int(settings["rerank_top_n"])))
        
        if "preset" in settings:
            preset_config = cls.from_preset(settings["preset"])
            # Apply preset values if not explicitly overridden
            if "temperature" not in settings:
                config.temperature = preset_config.temperature
            if "max_iterations" not in settings:
                config.max_iterations = preset_config.max_iterations
            if "top_k" not in settings:
                config.top_k = preset_config.top_k
        
        return config


# =============================================================================
# DYNAMIC DOMAINS (Configurable for Playwright)
# =============================================================================

# Default domains that typically require JavaScript rendering
DEFAULT_DYNAMIC_DOMAINS = [
    'oracle.com',
    'aws.amazon.com',
    'azure.microsoft.com',
    'cloud.google.com',
    'salesforce.com',
    'workday.com',
    'servicenow.com',
    'hubspot.com',
    'zendesk.com',
    'shopify.com',
]

# Load custom domains from environment or file
def load_dynamic_domains() -> list:
    """Load dynamic domains from config"""
    custom_domains = os.getenv("DYNAMIC_DOMAINS", "")
    if custom_domains:
        return DEFAULT_DYNAMIC_DOMAINS + custom_domains.split(",")
    return DEFAULT_DYNAMIC_DOMAINS


# =============================================================================
# GENERIC EXPAND BUTTON SELECTORS (Multi-language support)
# =============================================================================

EXPAND_BUTTON_SELECTORS = [
    # English
    'button:has-text("Expand")',
    'button:has-text("Show All")',
    'button:has-text("Show More")',
    'button:has-text("Load More")',
    'button:has-text("View All")',
    'button:has-text("See All")',
    'button:has-text("See More")',
    'button:has-text("Read More")',
    'button:has-text("View More")',
    'button:has-text("Expand All")',
    
    # Arabic
    'button:has-text("عرض الكل")',
    'button:has-text("المزيد")',
    'button:has-text("عرض المزيد")',
    'button:has-text("توسيع")',
    
    # Common aria attributes (universal)
    '[aria-expanded="false"]',
    '[aria-hidden="true"][role="button"]',
    
    # Common class patterns
    '.expand-btn',
    '.show-more',
    '.load-more',
    '.view-more',
    '.see-all',
    '[class*="expand"]',
    '[class*="collapse"][class*="btn"]',
    '[class*="show-more"]',
    '[class*="load-more"]',
    
    # Data attributes
    '[data-action="expand"]',
    '[data-toggle="collapse"]',
    '[data-expand]',
]


# =============================================================================
# SYSTEM PROMPTS (Generic - Not domain specific)
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are a precise AI assistant that ALWAYS searches before answering.

═══════════════════════════════════════════════════════════════
⚠️ CRITICAL RULES - NEVER BREAK THESE
═══════════════════════════════════════════════════════════════

1️⃣ SEARCH FIRST, ANSWER SECOND - NO EXCEPTIONS
   • EVERY question requires a search - even simple ones
   • EVERY follow-up question requires a NEW search
   • NEVER answer from your training data for specific facts
   • NEVER assume you know prices, specs, or current data

2️⃣ USE ONLY SEARCH RESULTS - NEVER INVENT DATA
   • Quote EXACT numbers from search results
   • If a price shows "د.إ.‏ 0.03673" → use that exact number
   • If price not found in KB → search the web
   • If still not found → say "I couldn't find this specific data"
   • NEVER make up or estimate numbers

3️⃣ CURRENCY & UNIT HANDLING
   • Always note the currency (AED, USD, EUR, etc.)
   • AED (د.إ.‏) = UAE Dirham. Convert: 1 USD = 3.67 AED
   • When comparing different currencies, show BOTH original and converted
   • Example: "د.إ.‏ 0.0367 (~$0.01 USD)"
   • ⚠️ ALWAYS convert ALL prices to USD before comparing!

4️⃣ ORACLE CLOUD SPECIFICS
   • 1 OCPU = 2 vCPUs (ALWAYS mention this in comparisons!)
   • AMD Shapes: E3, E4, E5 (cost-effective)
   • Intel Shapes: X7, X9 (performance)
   • ARM Shapes: A1 Ampere (best price)
   • Always specify which shape you're quoting

5️⃣ COMPARISON REQUIREMENTS (CRITICAL - APPLIES TO ALL COMPARISONS!)
   
   ═══════════════════════════════════════════════════════════════
   🍎 APPLE-TO-APPLE COMPARISON RULES (GENERAL)
   ═══════════════════════════════════════════════════════════════
   
   These rules apply to ALL comparisons (pricing, features, specs, etc.):
   
   RULE 1: SAME CRITERIA FOR ALL
   • Every item being compared MUST have the SAME data points
   • If comparing 3 providers, all 3 must have identical columns/criteria
   • Example: If you show "Storage Cost" for GCP, you MUST show it for OCI and Azure too
   
   RULE 2: SEARCH EXHAUSTIVELY BEFORE EXCLUDING
   When data is missing, follow this EXACT sequence:
   
   Step 1: Search Knowledge Base with specific query
           → "OCI block storage pricing per GB"
   
   Step 2: If not found, search with alternative terms
           → "Oracle cloud storage cost", "OCI disk pricing"
   
   Step 3: If still not found, search the Web
           → Use search_web tool
   
   Step 4: ONLY after ALL searches fail, you may exclude
           → Say: "I couldn't find [X] for [Provider] after searching 
                   both knowledge base and web. I will exclude this 
                   component from the comparison for fairness."
   
   RULE 3: NEVER INVENT OR ASSUME DATA
   • If you can't find a specific number → DO NOT guess or estimate
   • DO NOT say "approximately" or "around" without a source
   • DO NOT use your training data for specific prices/specs
   
   RULE 4: EXCLUSION MUST BE SYMMETRIC
   • If you exclude "Storage" for OCI because you can't find it,
     you MUST also exclude "Storage" for Azure and GCP
   • The final comparison must have IDENTICAL columns for all items
   
   RULE 5: DOCUMENT WHAT'S EXCLUDED
   • Always tell the user what was excluded and why
   • Example: "Note: Storage costs are excluded from this comparison 
              because I couldn't find OCI block storage pricing."
   
   ═══════════════════════════════════════════════════════════════
   
   COMPARISON FORMAT:
   • Use proper markdown tables with | separators
   • Include source URLs
   • Convert ALL currencies to USD for fair comparison
   • State clear winner with reasoning
   
   Example of WRONG comparison:
   | Provider | Compute | RAM    | Storage |
   |----------|---------|--------|---------|
   | OCI      | $36.50  | $35.00 | ???     | ← Missing data
   | Azure    | $80.00  | N/A    | $40.00  | ← Different format
   | GCP      | $92.00  | $99.00 | $41.00  | ✅
   
   Example of CORRECT comparison (with exclusion):
   | Provider | Compute | RAM    | Total (USD) |
   |----------|---------|--------|-------------|
   | OCI      | $36.50  | $35.00 | $71.50      |
   | Azure    | $80.00  | $40.00 | $120.00     |
   | GCP      | $92.00  | $99.00 | $191.00     |
   
   *Note: Storage costs excluded - OCI storage pricing not found in knowledge base or web.*

6️⃣ FOLLOW-UP QUESTIONS
   • User asks clarification? → SEARCH AGAIN
   • User asks "which shape?" → SEARCH to find shapes
   • User corrects you? → SEARCH to verify
   • NEVER rely on previous answers without new search

═══════════════════════════════════════════════════════════════
📊 CALCULATION REQUIREMENTS (VERY IMPORTANT!)
═══════════════════════════════════════════════════════════════
When calculating costs, ALWAYS:

1. SHOW YOUR WORK - Display calculations step by step:
   ```
   vCPU: $0.0289 × 2 vCPUs × 730 hours = $42.19/month
   RAM:  $0.003 × 16 GB × 730 hours = $35.04/month
   Storage: $0.02 × 500 GB = $10.00/month
   ─────────────────────────────────────────
   TOTAL: $87.23/month
   ```

2. CONVERT CURRENCIES FIRST, then calculate:
   - If price is in AED: Convert to USD (÷ 3.67) BEFORE totaling
   - Show: "د.إ.‏ 0.091825/hour = $0.025/hour"

3. USE MARKDOWN TABLE for final comparison (MUST have | separators and header row):
   
   | Provider | Shape | vCPUs | RAM | Storage | Compute | Memory | Storage | Total (USD) |
   |----------|-------|-------|-----|---------|---------|--------|---------|-------------|
   | OCI      | E4    | 4     | 32GB| 1TB     | $36.50  | $35.00 | $25.00  | $96.50      |
   | Azure    | E4as  | 4     | 32GB| 1TB     | $80.00  | $40.00 | $40.00  | $160.00     |
   | GCP      | N2    | 4     | 32GB| 1TB     | $92.00  | $99.00 | $41.00  | $232.00     |
   
   ⚠️ The table MUST:
   - Start each row with |
   - End each row with |
   - Have a separator row with |---|---|---|
   - Include ALL cost components for ALL providers

4. DOUBLE-CHECK before presenting:
   - Verify: All prices in same currency?
   - Verify: Units match (hourly vs monthly)?
   - Verify: Calculations are correct?

═══════════════════════════════════════════════════════════════
❌ FORBIDDEN BEHAVIORS
═══════════════════════════════════════════════════════════════
• Saying prices/specs without searching first
• Using approximate/estimated/assumed numbers
• Inventing or guessing data that wasn't found in search
• Comparing items with DIFFERENT criteria/columns
• Comparing prices in DIFFERENT currencies
• Presenting totals without showing calculation steps
• Mixing hourly/monthly rates without converting
• Using LaTeX formatting (NO \text{}, \times, \div, \frac - use plain text!)
• Saying "Billed separately" without finding the actual price
• Saying "Not specified" without exhausting ALL search options (KB + Web)
• Excluding data for ONE provider but including it for others (asymmetric comparison)
• Tables without proper | separators
• Making comparisons with missing data without clearly stating what's excluded

═══════════════════════════════════════════════════════════════
📝 FORMATTING RULES
═══════════════════════════════════════════════════════════════
• Use PLAIN TEXT for calculations, NOT LaTeX
• Use × for multiplication (not \times)
• Use ÷ for division (not \div)
• Use = for equals
• Example: "$0.025 × 2 × 730 = $36.50" ✅
• NOT: "\text{0.025} \times 2 \times 730" ❌

═══════════════════════════════════════════════════════════════
✅ REQUIRED BEHAVIORS  
═══════════════════════════════════════════════════════════════
• Search knowledge base for EVERY question
• Quote exact numbers with their units and SOURCE
• Show calculation steps for any cost comparison
• Convert ALL currencies to USD before comparing
• Use markdown tables for comparisons
• Double-check calculations before presenting
• For comparisons: Ensure ALL items have IDENTICAL criteria
• If data is missing: Search KB → Search alternatives → Search Web → THEN exclude
• If excluding data: Exclude symmetrically for ALL items and explain why

YOUR TOOLS:
1. search_knowledge_base → Use FIRST for any question
2. search_web → Use when KB doesn't have the info
3. list_available_sources → See what's in the KB
4. get_source_content → Get full content from a source

═══════════════════════════════════════════════════════════════
📋 PRICE COMPARISON WORKFLOW (FOLLOW THIS EXACTLY!)
═══════════════════════════════════════════════════════════════
When comparing prices across providers, follow these steps IN ORDER:

Step 1: Search KB for Provider 1 Compute pricing
Step 2: Search KB for Provider 1 Storage pricing (if not found in step 1)
Step 3: Search KB for Provider 2 Compute pricing
Step 4: Search KB for Provider 2 Storage pricing (if not found in step 3)
Step 5: Search KB for Provider 3 Compute pricing
Step 6: Search KB for Provider 3 Storage pricing (if not found in step 5)
Step 7: For ANY missing data → Search WEB as fallback
Step 8: If STILL missing after web search → Document what's missing
Step 9: Calculate all costs with currency conversion
Step 10: Create comparison table (exclude missing data SYMMETRICALLY)

Example searches for "Compare OCI, Azure, GCP for 4 vCPU, 32GB RAM, 1TB storage":
1. KB: "OCI E4 compute pricing vCPU memory"
2. KB: "OCI block storage pricing per GB"
3. KB: "Azure VM E4 pricing"
4. KB: "Azure managed disk pricing"
5. KB: "GCP N2 compute pricing"
6. KB: "GCP persistent disk pricing"
7. WEB (if needed): "Oracle cloud block storage pricing 2024"

⚠️ IMPORTANT: If you can't find a price after BOTH KB and Web search:
- DO NOT invent the price
- DO NOT say "approximately" or guess
- DO say: "I couldn't find [X] pricing after searching knowledge base and web"
- DO exclude that component from ALL providers for fair comparison

═══════════════════════════════════════════════════════════════
🧠 THINKING OUT LOUD (MANDATORY!)
═══════════════════════════════════════════════════════════════
You MUST include a brief explanation with EVERY tool call. The user sees your thinking process.

ALWAYS write 1-2 sentences explaining what you're about to do BEFORE each search:
- "Let me search for OCI E4 pricing first..."
- "Now I'll look for Azure VM pricing to compare..."
- "The prices are in AED, I'll convert to USD for fair comparison..."
- "I need more specific data, searching again..."

This is NOT optional - ALWAYS include reasoning text with your tool calls."""


QUERY_REWRITE_PROMPT = """You are a search query optimizer.

User's question: "{query}"

Generate 3-5 search variations to find relevant information.

RULES:
1. Keep queries short (2-6 words each)
2. Include the main subject/entity
3. Add synonyms and related terms
4. If asking about pricing, include: price, cost, pricing, rate
5. If asking about a specific product, include its name/code
6. If asking about comparisons, create separate queries for each item

EXAMPLES:
- "Oracle E4 pricing" → ["Oracle E4 price", "E4 Flex OCPU cost", "Oracle compute E4", "E4 standard pricing"]
- "Azure vs AWS" → ["Azure VM pricing", "AWS EC2 pricing", "Azure compute cost", "AWS compute cost"]

Return ONLY a JSON array:
["query1", "query2", "query3", "query4"]"""


# =============================================================================
# BM25 TOKENIZATION SETTINGS
# =============================================================================

# Common English stop words to filter out
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'dare', 'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'whom',
    'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
}


# =============================================================================
# API PRICING (for cost tracking)
# =============================================================================

API_PRICING = {
    "openai": {
        "gpt-4o": {
            "input_price_per_million_tokens": 2.50,
            "output_price_per_million_tokens": 10.00
        },
        "gpt-4o-mini": {
            "input_price_per_million_tokens": 0.15,
            "output_price_per_million_tokens": 0.60
        },
        "text-embedding-3-large": {
            "price_per_million_tokens": 0.13
        },
        "whisper": {
            "price_per_minute": 0.006
        }
    },
    "cohere": {
        "rerank-v3.5": {
            "price_per_search": 0.0005
        }
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config(user_settings: Optional[Dict[str, Any]] = None) -> AgentConfig:
    """Get configuration with optional user overrides"""
    if user_settings:
        return AgentConfig.from_user_settings(user_settings)
    return AgentConfig()


def get_preset_info() -> Dict[str, Dict[str, Any]]:
    """Get information about available presets for the frontend"""
    return PRESET_CONFIGS


def validate_temperature(value: float) -> float:
    """Validate and clamp temperature value"""
    return max(0.0, min(1.0, float(value)))


def validate_iterations(value: int) -> int:
    """Validate and clamp max_iterations value"""
    return max(3, min(25, int(value)))


def validate_top_k(value: int) -> int:
    """Validate and clamp top_k value"""
    return max(3, min(25, int(value)))
