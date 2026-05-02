### store_page_score
Measures overall quality and completeness of a game’s Steam store page.  
- Combines visuals (screenshots, trailer), written content (detailed description), accessibility (languages), and credibility (website, support email).  
- Heavily weighted toward screenshots (30%) and trailer presence (25%) since these strongly influence first impressions.  
- Uses caps (e.g., max 10 screenshots, 20 languages) to prevent extreme values from dominating.

---

### platform_reach
Represents how widely the game is available across platforms.  
- Computed as `platform_count / 3.0` assuming up to 3 major platforms.  
- Higher values indicate broader accessibility and potential audience size.

---

### is_mature_content
Binary flag indicating whether the game targets mature audiences.  
- `True` if `required_age >= 17`, otherwise `0`.  
- Included because age-restricted games typically have a smaller addressable market.

---

### marketing_score
Proxy for marketing effort and visibility.  
- Combines presence of an official website (30%) and visual assets (screenshots, 70%).  
- Assumes more screenshots and a website correlate with stronger promotion.

---

### publisher_backing
Estimates the level of publisher support behind the game.  
- Based on whether a publisher exists (60%) and how many publishers are involved (40%, capped at 3).  
- Higher values suggest more resources, funding, or distribution support.

---

### localization_score
Captures how well the game is adapted for global audiences.  
- Based on supported languages (70%) and full audio languages (30%).  
- Reflects both accessibility (text) and deeper localization (voice).

---

### steam_integration
Measures how well the game utilizes Steam platform features.  
- Includes achievements, trading cards, cloud saves, workshop, controller support, and family sharing.  
- Workshop (20%) and achievements (25%) are weighted higher due to stronger engagement impact.  
- Reflects player retention and ecosystem engagement rather than direct marketing.
