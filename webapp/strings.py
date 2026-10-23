"""Text for the scan and check screens in English, Hindi and Telugu.
Hindi and Telugu were machine-drafted: have a native speaker check them before real use."""

LANGS = {"en": "English", "hi": "हिन्दी", "te": "తెలుగు"}

T = {
    # ---------------------------------------------------------------- scan screen
    "scan_title": {"en": "Scan a vial", "hi": "शीशी स्कैन करें", "te": "సీసాను స్కాన్ చేయండి"},
    "check_title": {"en": "Check a vaccine", "hi": "टीका जाँचें", "te": "టీకాను తనిఖీ చేయండి"},
    "scan_hint": {
        "en": "Photograph the VVM (the circle with a square inside) up close, in good light, with it in the middle of the frame.",
        "hi": "VVM (गोले के अंदर चौकोर) की पास से, अच्छी रोशनी में, फ़्रेम के बीच में रखकर फ़ोटो लें।",
        "te": "VVM (వృత్తం లోపల చతురస్రం) ను దగ్గరగా, మంచి వెలుతురులో, ఫ్రేమ్ మధ్యలో ఉంచి ఫోటో తీయండి."},
    "tab_photo": {"en": "Photo", "hi": "फ़ोटो", "te": "ఫోటో"},
    "tab_camera": {"en": "Live camera", "hi": "लाइव कैमरा", "te": "లైవ్ కెమెరా"},
    "upload": {"en": "Take or choose a photo", "hi": "फ़ोटो लें या चुनें", "te": "ఫోటో తీయండి లేదా ఎంచుకోండి"},
    "upload_many": {"en": "Take or choose photos (one VVM per photo)",
                    "hi": "फ़ोटो लें या चुनें (हर फ़ोटो में एक VVM)",
                    "te": "ఫోటోలు తీయండి లేదా ఎంచుకోండి (ఒక్కో ఫోటోలో ఒక VVM)"},
    "camera": {"en": "Point at the VVM and tap to capture", "hi": "VVM की ओर कैमरा रखें और फ़ोटो लें",
               "te": "కెమెరాను VVM వైపు ఉంచి ఫోటో తీయండి"},
    "heat_used": {"en": "Heat allowance used", "hi": "गर्मी की सीमा का उपयोग", "te": "వాడిన వేడి పరిమితి"},

    # ---------------------------------------------------------------- verdicts (health worker)
    "USE": {"en": "Use this vial", "hi": "इस शीशी का उपयोग करें", "te": "ఈ సీసాను వాడండి"},
    "USE_sub": {
        "en": "The square is lighter than the circle. Heat exposure is within the limit.",
        "hi": "चौकोर, गोले से हल्का है। गर्मी का असर सीमा के भीतर है।",
        "te": "చతురస్రం వృత్తం కంటే లేతగా ఉంది. వేడి ప్రభావం పరిమితిలోనే ఉంది."},
    "DISCARD": {"en": "Do not use", "hi": "उपयोग न करें", "te": "వాడవద్దు"},
    "DISCARD_sub": {
        "en": "The square matches or is darker than the circle: the heat limit is reached. Set the vial aside, mark it “Do not use” and report it as your programme requires.",
        "hi": "चौकोर, गोले जैसा या उससे गहरा है: गर्मी की सीमा पूरी हो चुकी है। शीशी अलग रखें, उस पर “उपयोग न करें” लिखें और नियम के अनुसार रिपोर्ट करें।",
        "te": "చతురస్రం వృత్తంతో సమానంగా లేదా ముదురుగా ఉంది: వేడి పరిమితి చేరుకుంది. సీసాను పక్కన పెట్టి, “వాడవద్దు” అని గుర్తు పెట్టి, నియమాల ప్రకారం నివేదించండి."},
    "RE-CHECK": {"en": "Check again by eye", "hi": "आँख से दोबारा जाँचें", "te": "కంటితో మళ్ళీ తనిఖీ చేయండి"},
    "RE-CHECK_near_threshold": {
        "en": "This reading is very close to the discard point. Look at the label in good light: if the square is not clearly lighter than the circle, do not use the vial.",
        "hi": "यह रीडिंग फेंकने की सीमा के बहुत पास है। अच्छी रोशनी में लेबल देखें: अगर चौकोर साफ़ तौर पर गोले से हल्का नहीं है, तो शीशी का उपयोग न करें।",
        "te": "ఈ రీడింగ్ పారవేసే పరిమితికి చాలా దగ్గరగా ఉంది. మంచి వెలుతురులో లేబుల్ చూడండి: చతురస్రం వృత్తం కంటే స్పష్టంగా లేతగా లేకపోతే, సీసాను వాడవద్దు."},
    "RE-CHECK_disagree": {
        "en": "The two reading methods disagree on this photo. Look at the label in good light: if the square is not clearly lighter than the circle, do not use the vial.",
        "hi": "इस फ़ोटो पर दोनों तरीकों का नतीजा अलग है। अच्छी रोशनी में लेबल देखें: अगर चौकोर साफ़ तौर पर गोले से हल्का नहीं है, तो शीशी का उपयोग न करें।",
        "te": "ఈ ఫోటోపై రెండు పద్ధతుల ఫలితాలు వేరుగా ఉన్నాయి. మంచి వెలుతురులో లేబుల్ చూడండి: చతురస్రం వృత్తం కంటే స్పష్టంగా లేతగా లేకపోతే, సీసాను వాడవద్దు."},
    "RETAKE": {"en": "Take the photo again", "hi": "फ़ोटो दोबारा लें", "te": "ఫోటో మళ్ళీ తీయండి"},

    # ---------------------------------------------------------------- verdicts (individual)
    "p_USE": {"en": "Heat indicator looks OK", "hi": "गर्मी सूचक ठीक दिख रहा है", "te": "వేడి సూచిక సరిగ్గా ఉంది"},
    "p_USE_sub": {
        "en": "The indicator shows this vaccine has not had too much heat. The health worker should still check the expiry date on the label.",
        "hi": "सूचक दिखाता है कि इस टीके पर ज़्यादा गर्मी का असर नहीं हुआ है। स्वास्थ्यकर्मी को लेबल पर समाप्ति तिथि भी देखनी चाहिए।",
        "te": "ఈ టీకాకు ఎక్కువ వేడి తగలలేదని సూచిక చూపిస్తోంది. ఆరోగ్య కార్యకర్త లేబుల్‌పై గడువు తేదీని కూడా చూడాలి."},
    "p_DISCARD": {"en": "Indicator shows too much heat", "hi": "सूचक ज़्यादा गर्मी दिखा रहा है",
                  "te": "సూచిక ఎక్కువ వేడిని చూపిస్తోంది"},
    "p_DISCARD_sub": {
        "en": "This vial may have lost strength. Politely ask the health worker to look at the label and use another vial.",
        "hi": "इस शीशी का असर कम हो सकता है। विनम्रता से स्वास्थ्यकर्मी से लेबल देखने और दूसरी शीशी इस्तेमाल करने को कहें।",
        "te": "ఈ సీసా ప్రభావం తగ్గి ఉండవచ్చు. లేబుల్ చూసి వేరే సీసా వాడమని ఆరోగ్య కార్యకర్తను మర్యాదగా అడగండి."},
    "p_RE-CHECK": {"en": "Can't be sure", "hi": "पक्का नहीं कह सकते", "te": "ఖచ్చితంగా చెప్పలేము"},
    "p_RE-CHECK_sub": {
        "en": "The indicator is close to its limit. Ask the health worker to look at the label in good light before using this vial.",
        "hi": "सूचक अपनी सीमा के पास है। इस शीशी के इस्तेमाल से पहले स्वास्थ्यकर्मी से अच्छी रोशनी में लेबल देखने को कहें।",
        "te": "సూచిక తన పరిమితికి దగ్గరగా ఉంది. ఈ సీసా వాడే ముందు మంచి వెలుతురులో లేబుల్ చూడమని ఆరోగ్య కార్యకర్తను అడగండి."},

    # ---------------------------------------------------------------- retake reasons
    "why_not_found": {"en": "Couldn't find the VVM. Fill the frame with the label and keep the circle in the middle.",
                      "hi": "VVM नहीं मिला। लेबल को फ़्रेम में भरें और गोले को बीच में रखें।",
                      "te": "VVM కనిపించలేదు. లేబుల్‌తో ఫ్రేమ్ నింపి, వృత్తాన్ని మధ్యలో ఉంచండి."},
    "why_too_small": {"en": "The VVM is too small in the photo. Move closer.",
                      "hi": "फ़ोटो में VVM बहुत छोटा है। पास जाएँ।",
                      "te": "ఫోటోలో VVM చాలా చిన్నగా ఉంది. దగ్గరగా వెళ్ళండి."},
    "why_blurry": {"en": "The photo is blurry. Hold the phone steady and tap the screen on the VVM to focus.",
                   "hi": "फ़ोटो धुंधली है। फ़ोन स्थिर रखें और फ़ोकस के लिए स्क्रीन पर VVM को टैप करें।",
                   "te": "ఫోటో మసకగా ఉంది. ఫోన్‌ను స్థిరంగా పట్టుకుని, ఫోకస్ కోసం స్క్రీన్‌పై VVM ను తాకండి."},
    "why_too_dark": {"en": "Too dark. Move to better light.", "hi": "बहुत अंधेरा है। अच्छी रोशनी में जाएँ।",
                     "te": "చాలా చీకటిగా ఉంది. మంచి వెలుతురులోకి వెళ్ళండి."},
    "why_glare": {"en": "Light is reflecting off the label. Tilt the vial slightly or move out of direct light.",
                  "hi": "लेबल पर रोशनी चमक रही है। शीशी थोड़ा झुकाएँ या सीधी रोशनी से हटें।",
                  "te": "లేబుల్‌పై వెలుతురు మెరుస్తోంది. సీసాను కొంచెం వంచండి లేదా నేరుగా పడే వెలుతురు నుండి దూరంగా జరగండి."},
    "why_tilted": {"en": "The label is at a steep angle. Hold the camera straight over it.",
                   "hi": "लेबल बहुत तिरछा है। कैमरा उसके ठीक ऊपर सीधा रखें।",
                   "te": "లేబుల్ బాగా వంగి ఉంది. కెమెరాను దానికి నేరుగా పైన పట్టుకోండి."},
}


def t(key, lang="en"):
    d = T.get(key)
    if d is None:
        return key
    return d.get(lang) or d["en"]
