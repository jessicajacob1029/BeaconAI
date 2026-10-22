## Author review of the global view

**This is an author review, not a clinical validation.** It was written by the project author from general medical knowledge, without clinician sign-off, and must not be read as evidence that the model reasons correctly in a clinical sense. It checks whether each condition's top features (in the tables above) look like the presenting symptoms and risk factors one would expect, and labels each surprise as expected, plausible, or odd, with a likely cause.

### Summary

- Six of the eight conditions are led by textbook features: pulmonary embolism, unstable angina, anaphylaxis, croup, NSTEMI/STEMI and pneumonia (for its respiratory items). The model's strongest reasons are mostly the ones a clinician would look for.
- The automatic flags caught only the croup age buckets (expected). They cannot catch the main oddity below: a feature that is genuinely common with a condition in DDXPlus, even though it is not common in real medicine. Such a feature passes every prevalence check.
- **Main oddity: skin questions.** "Do you have any lesions, redness or problems on your skin...", "What color is the rash? = pink" and "Do your lesions peel off?" rank among the top features for pneumonia (99.3% of DDXPlus pneumonia patients answer yes to the skin question) and influenza (86.5%). This looks like a quirk of how DDXPlus generated its synthetic patients, not a model error: the model is faithful to its data. It matters for users, though: an explanation such as "pink rash" for pneumonia would confuse them and undermine trust. Worth handling before explanations are shown (Phase 5 or 10), for example by reviewing which features may appear in user-facing reasons.

### By condition

| Condition | Verdict | Notes |
|---|---|---|
| Pneumonia | Mostly expected, one odd group | Expected: coloured sputum, chills, coughing up blood, pain on deep breathing, loss of appetite, fatigue, symptoms worse at night. **Odd:** the skin lesion and pink rash items (synthetic-data quirk, see above). |
| Possible NSTEMI / STEMI | Plausible, one odd | Expected: sweating, "scary" or "sickening" pain, nausea, diabetes, former smoker, poor circulation, fatigue. Chest pain itself is not in the top 10. That is plausible, because chest pain is shared with angina and other chest conditions and so does not separate them. **Odd:** HIV infection in 67% of these patients. HIV is a real cardiovascular risk factor, but this prevalence is far above any real population; a synthetic-data quirk. Loss of appetite is weak but plausible. |
| Unstable angina | Expected, one odd | Textbook: worsening over the last 2 weeks with less exertion, chest pain at rest, symptoms with exertion relieved by rest, previous heart attack or angina, diabetes, family history of early cardiovascular disease, sweating. **Odd:** "Do you exercise regularly, 4 times per week or more?" pushes toward unstable angina (66% of these patients say yes). Regular exercise is protective in real medicine, so this is probably a data-generation quirk or a question whose meaning differs from its wording. |
| Pulmonary embolism | Expected | Textbook risk factors and signs (close to the Wells criteria): swelling, recent surgery, active cancer, immobility for more than 3 days, previous deep vein thrombosis, coughing up blood, pain on deep breathing, fainting. Pain at the shoulder blades is plausible as referred pleuritic pain. |
| Anaphylaxis | Expected | Textbook: contact with a known allergen, severe food allergy, swelling, skin rash, high-pitched sound when breathing in (stridor), wheezing, light-headedness, fainting. The skin items are clinically right here. "Lesions do not peel off" is a neutral answer to a skin question. |
| Influenza | Mostly expected, one odd group | Expected: exhaustion, chills, widespread muscle pain, sore throat, sweating, loss of appetite. Being immunosuppressed is plausible as a risk factor. Fever itself is not in the top 10. That is plausible, because fever is common to many infections and so does not separate them. **Odd:** the pink rash and skin lesion items (86.5% prevalence; same quirk as pneumonia). |
| Acute rhinosinusitis | Expected core, some chronic features | Expected: green or yellow nasal discharge, loss of smell, a cold in the last 2 weeks. **Plausible but telling:** nasal polyps (91% of these patients), deviated septum, allergies and asthma are classic *chronic* rhinosinusitis risk factors. That overlap explains the main Phase 2 confusion (acute predicted as chronic, 13%). **Odd:** "burning" pain, a history of pneumonia and reflux disease are weak or unexpected for acute sinusitis. |
| Croup | Expected, one odd | Textbook: barking ("whooping") cough, symptoms worse at night, high-pitched sound when breathing in (stridor), family history of croup, runny nose, fever, and young age (the only automatic flags: age 1 to 4 and under 1, both expected). **Odd:** producing more saliva than usual is a classic sign of epiglottitis, not croup; possibly the DDXPlus definitions overlap there. Vaccinations being up to date is neutral. |

### What this means

- No feature suggests leakage or a broken model. The surprises trace back to how DDXPlus was generated, and the model reproduces them faithfully.
- The skin-question quirk affects explanation quality more than accuracy. Users should not be shown "pink rash" as a reason for pneumonia without review.
- The automatic flags are useful for catching collinearity and rare features, but they cannot replace a human (ideally clinical) review, because they only compare the model with its own data.

## Reviewer notes
