# Composite Features

### store_page_score
Measures overall quality and completeness of a game's Steam store page.
- Combines visual assets (screenshots, 40%), written content (detailed description, 25%), language accessibility (weighted language score, 20%), and credibility signals (official website, 10%; support email, 5%).
- Screenshot count is capped at 10 to prevent extreme values from dominating.
- `has_detailed_desc` is automatically derived: it is set to 1 if `about_length` exceeds 500 characters, otherwise 0.

---

### platform_reach
Represents how widely the game is available across platforms.
- Computed as `platform_count / 3.0`, assuming up to 3 major platforms (Windows, Mac, Linux).
- Higher values indicate broader accessibility and potential audience size.

---

### is_mature_content
Binary flag indicating whether the game targets mature audiences.
- Set to 1 if `required_age >= 17`, otherwise 0.
- Included because age-restricted games typically have a smaller addressable market.

---

### marketing_score
Proxy for a game's marketing effort and external visibility.
- Combines visual assets (screenshots, 45%), official website presence (35%), and support email presence (20%).
- Screenshot count is capped at 10.
- Trailer presence is no longer included in v2 as it was found to carry no predictive signal in the training data.

---

### localization_score
Captures how well the game is adapted for global audiences.
- Weighted combination of `weighted_language_score` (75%) and full audio language coverage (25%, capped at 10 languages).
- `weighted_language_score` accounts for which languages are supported, not just how many, reflecting the relative market size of each language on Steam.

---

### steam_integration
Measures how deeply the game utilizes Steam platform features.
- Combines Steam Achievements (35%), Cloud Save (25%), Controller Support (25%), and Family Sharing (15%).
- Trading cards and Workshop are no longer included in v2 as both were post-success indicators rather than pre-launch decisions.
- Reflects player retention and ecosystem engagement potential.
