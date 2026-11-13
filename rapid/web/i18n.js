// i18n.js — English / Greek.
//
//  * UI text is keyed by its English wording: t('Update forecast') returns
//    the Greek when Greek is selected.  {name} placeholders are filled from
//    the second argument.  Static HTML uses data-i18n / data-i18n-title /
//    data-i18n-ph attributes and is re-translated on every language switch.
//  * Sentences composed by the C++ engine (actions, advice, confidence
//    reasons, calibration, warnings, errors, status) are translated by
//    pattern in tr().  Anything not covered stays in English rather than
//    breaking, so new engine wording degrades gracefully.
'use strict';

const I18N = (() => {
  let lang = 'en';
  try { lang = localStorage.getItem('wilson-lang') === 'el' ? 'el' : 'en'; } catch (_) {}

  // ── UI strings ──
  const EL = {
    // top bar
    'Fire spread forecast': 'Πρόγνωση εξάπλωσης πυρκαγιάς',
    'Coordinates (38.56, 21.94) or place name': 'Συντεταγμένες (38.56, 21.94) ή τοπωνύμιο',
    'Locate': 'Εντοπισμός',
    'Forecast window': 'Ορίζοντας πρόγνωσης',
    'Apply': 'Εφαρμογή',
    'Update forecast': 'Ενημέρωση πρόγνωσης',
    'New incident': 'Νέο συμβάν',
    'Confidence': 'Αξιοπιστία',
    'Dark': 'Σκούρο', 'Light': 'Ανοιχτό', 'Dark theme': 'Σκούρο θέμα',
    'Switch to the light theme': 'Αλλαγή σε ανοιχτό θέμα', 'Switch to the dark theme': 'Αλλαγή σε σκούρο θέμα',
    'Re-run the forecast for the current time with fresh weather': 'Νέος υπολογισμός για την τρέχουσα ώρα με νέα μετεωρολογικά δεδομένα',
    'Close this incident and start a new one': 'Κλείσιμο του συμβάντος και έναρξη νέου',
    'How far to trust this forecast': 'Πόσο αξιόπιστη είναι η πρόγνωση',
    'Incident {pos}': 'Συμβάν {pos}',
    'Ignition {t0} · forecast to {t1}': 'Έναρξη {t0} · πρόγνωση έως {t1}',
    // reference panels
    'Weather': 'Καιρός', 'Forecast quality': 'Αξιοπιστία πρόγνωσης', 'Close': 'Κλείσιμο',
    'Wind': 'Άνεμος', 'From': 'Από', 'Temperature': 'Θερμοκρασία', 'Rel. humidity': 'Σχετ. υγρασία',
    'Source: {s}': 'Πηγή: {s}', 'Time': 'Ώρα', 'km/h': 'km/h', 'RH %': 'ΣΥ %',
    'Overall': 'Συνολικά',
    'Terrain': 'Ανάγλυφο', 'Vegetation': 'Βλάστηση', 'Places': 'Οικισμοί', 'Field data': 'Δεδομένα πεδίου',
    '{n} report(s)': '{n} αναφορές', 'Surface wind': 'Επιφανειακός άνεμος', 'Computed': 'Υπολογισμός',
    '{ms} ms, {n} scenarios, {c} m cells': '{ms} ms, {n} σενάρια, κελιά {c} m',
    // left column
    'Fire forecast': 'Πρόγνωση πυρκαγιάς', 'Widen': 'Διεύρυνση', 'Narrow': 'Στένωση', 'Hide': 'Απόκρυψη',
    'Widen the column': 'Διεύρυνση στήλης', 'Narrow the column': 'Στένωση στήλης', 'Expand the operations panel': 'Ανάπτυξη πίνακα επιχειρήσεων', 'Minimize the operations panel': 'Ελαχιστοποίηση πίνακα επιχειρήσεων', 'Collapse the column to the left': 'Σύμπτυξη της στήλης αριστερά',
    'Show the fire forecast column': 'Εμφάνιση της στήλης πρόγνωσης',
    'Fire behaviour': 'Συμπεριφορά πυρκαγιάς', 'Head of fire moving': 'Κατεύθυνση μετώπου',
    'Towards {dir} ({deg}°)': 'Προς τα {dir} ({deg}°)',
    'Forecast window {t0} to {t1}': 'Πρόγνωση {t0} έως {t1}',
    'Head spread': 'Ταχύτητα μετώπου', 'Flame length': 'Μήκος φλόγας', 'Head reach': 'Απόσταση μετώπου',
    'Burned now': 'Καμένα τώρα', 'New growth': 'Νέα εξάπλωση', 'Area at end': 'Έκταση στο τέλος',
    'Area danger class: {c}': 'Κατηγορία κινδύνου περιοχής: {c}', 'Danger map': 'Χάρτης κινδύνου',
    'Show the danger map': 'Εμφάνιση χάρτη κινδύνου',
    'Map layer': 'Επίπεδο χάρτη', 'Most likely': 'Πιθανότερο', 'Probability': 'Πιθανότητα', 'Danger': 'Κίνδυνος',
    'Fire by': 'Φωτιά έως', 'Scenarios': 'Σενάρια', 'Wide uncertainty (wind or fuel unclear)': 'Ευρεία αβεβαιότητα (ασαφής άνεμος ή καύσιμη ύλη)',
    'Surface wind field (terrain-adjusted)': 'Πεδίο επιφανειακού ανέμου (με βάση το ανάγλυφο)',
    'Share of the {n} scenarios in which fire reaches each cell.': 'Ποσοστό των {n} σεναρίων στα οποία η φωτιά φτάνει σε κάθε σημείο.',
    ' Wide uncertainty is on.': ' Η ευρεία αβεβαιότητα είναι ενεργή.',
    'With the plan: about {ha} ha less burned by {t}.': 'Με το σχέδιο: περίπου {ha} ha λιγότερα καμένα έως τις {t}.',
    'The plan does not materially reduce the burned area. See places at risk.': 'Το σχέδιο δεν μειώνει ουσιαστικά την καμένη έκταση. Δείτε τις απειλούμενες θέσεις.',
    'Places at risk': 'Απειλούμενες θέσεις', 'Place': 'Θέση', 'Arrival': 'Άφιξη', 'Prob.': 'Πιθ.', 'Conf.': 'Αξιοπ.',
    'Burning': 'Καίγεται', 'Not expected': 'Δεν αναμένεται', 'in {d}': 'σε {d}', 'With plan: {s}': 'Με σχέδιο: {s}',
    'River': 'Ποτάμι', 'Road': 'Δρόμος', 'Marked asset': 'Σημειωμένη θέση',
    'No settlements or marked assets reached within the {h} h window. Click any point on the map for its expected arrival time.':
      'Κανένας οικισμός ή σημειωμένη θέση δεν απειλείται μέσα στο διάστημα {h} ωρών. Επιλέξτε οποιοδήποτε σημείο στον χάρτη για την αναμενόμενη ώρα άφιξης.',
    'High': 'Υψηλή', 'Medium': 'Μέτρια', 'Low': 'Χαμηλή',
    // danger
    'Danger class': 'Κατηγορία κινδύνου', '1-hour exposure zones': 'Ζώνες έκθεσης 1 ώρας',
    'Fire area is in danger class {c}. Danger combines potential spread rate with travel time to settlements and marked assets. Peak at {t}.':
      'Η περιοχή της φωτιάς είναι σε κατηγορία κινδύνου {c}. Ο κίνδυνος συνδυάζει τη δυνητική ταχύτητα εξάπλωσης με τον χρόνο άφιξης σε οικισμούς και σημειωμένες θέσεις. Μέγιστο στις {t}.',
    '{name}: fire within {km} km to the {dir} reaches it in under 1 h': '{name}: φωτιά σε απόσταση έως {km} km προς τα {dir} φτάνει σε λιγότερο από 1 ώρα',
    '({ha} ha, spread up to {v} km/h)': '({ha} ha, εξάπλωση έως {v} km/h)',
    'No settlement can be reached within 1 h from its surroundings.': 'Κανένας οικισμός δεν μπορεί να απειληθεί μέσα σε 1 ώρα από τον περίγυρό του.',
    // legend
    'Fire arrival': 'Άφιξη φωτιάς', 'Burned': 'Καμένα', 'Within 1 h': 'Εντός 1 ώρας', '1 to 2 h': '1 έως 2 ώρες',
    '2 to 3 h': '2 έως 3 ώρες', 'Later': 'Αργότερα', 'Possible (pessimistic)': 'Πιθανή (δυσμενές σενάριο)',
    // dock
    'Operations': 'Επιχειρήσεις', 'Minimize': 'Ελαχιστοποίηση', 'Expand': 'Ανάπτυξη',
    'Without plan': 'Χωρίς σχέδιο', 'With plan': 'Με σχέδιο',
    'Show the forecast without or with the applied actions': 'Πρόγνωση χωρίς ή με τις ενέργειες που εφαρμόστηκαν',
    'Forecast with the applied actions': 'Πρόγνωση με τις ενέργειες που εφαρμόστηκαν',
    'No actions applied yet. Switch on an action or draw your own.': 'Δεν έχουν εφαρμοστεί ενέργειες. Ενεργοποιήστε μια ενέργεια ή σχεδιάστε δική σας.',
    'Resources available': 'Διαθέσιμα μέσα', 'Aircraft': 'Εναέρια', 'Engines': 'Οχήματα', 'Dozers': 'Χωματουργικά',
    '{n} assigned': '{n} σε χρήση',
    'Field reports': 'Αναφορές πεδίου', 'Burned area': 'Καμένη έκταση', 'Measured wind': 'Μετρημένος άνεμος', 'Protect asset': 'Προστασία θέσης',
    'Own actions': 'Δικές ενέργειες', 'Firebreak': 'Αντιπυρική ζώνη', 'Air drop': 'Ρίψη νερού', 'Engine': 'Όχημα', 'Evacuation': 'Εκκένωση',
    'Draw the area that has burned; the model recalibrates to it': 'Σχεδιάστε την καμένη έκταση· το μοντέλο βαθμονομείται σε αυτή',
    'Enter wind measured on site': 'Καταχώριση ανέμου μετρημένου επί τόπου',
    'Mark an asset to protect and get its time to impact': 'Σημειώστε θέση προς προστασία για να δείτε τον χρόνο άφιξης',
    'Draw a firebreak / dozer line': 'Σχεδιάστε αντιπυρική ζώνη', 'Place a water / retardant drop line': 'Τοποθέτηση γραμμής ρίψης νερού / επιβραδυντικού',
    'Place an engine': 'Τοποθέτηση οχήματος',
    'Action plan': 'Σχέδιο δράσης', 'Recommended actions': 'Προτεινόμενες ενέργειες', 'Other options': 'Άλλες επιλογές',
    'Applied': 'Σε εφαρμογή', 'Recommended': 'Προτείνεται', 'Advisory': 'Ενημέρωση', 'Own': 'Δική σας', 'Remove': 'Αφαίρεση',
    'Immediately': 'Άμεσα', 'Immediately (before {t})': 'Άμεσα (πριν τις {t})', 'Start before {t}': 'Έναρξη πριν τις {t}',
    '{p}% of forecasts': '{p}% των προγνώσεων', 'Effective in {p}% of scenarios': 'Αποτελεσματική στο {p}% των σεναρίων',
    'Requires an additional {r}': 'Απαιτεί επιπλέον {r}', 'resource': 'μέσο',
    'aircraft': 'εναέριο μέσο', 'engine': 'όχημα', 'dozer': 'χωματουργικό',
    'Remove from the plan': 'Αφαίρεση από το σχέδιο', 'Apply this action': 'Εφαρμογή της ενέργειας',
    'Click to copy': 'Αντιγραφή', 'Own {k}': 'Δική σας ενέργεια: {k}', 'Active from {t}': 'Ενεργή από τις {t}',
    'No actions applied.': 'Δεν έχουν εφαρμοστεί ενέργειες.',
    'Switch on a recommended action, or draw your own.': 'Ενεργοποιήστε μια προτεινόμενη ενέργεια ή σχεδιάστε δική σας.',
    'Enter available resources for recommendations, or draw your own.': 'Καταχωρίστε τα διαθέσιμα μέσα για προτάσεις ή σχεδιάστε δικές σας ενέργειες.',
    '{n} applied': '{n} σε εφαρμογή', ' (about {ha} ha saved)': ' (περίπου {ha} ha λιγότερα)',
    ', {r} recommended, {o} other options': ', {r} προτεινόμενες, {o} άλλες επιλογές',
    'Request: {s}': 'Αίτημα: {s}',
    // prompts and dialogs
    'Select the ignition point on the map, or enter coordinates above.': 'Επιλέξτε το σημείο έναρξης στον χάρτη ή εισαγάγετε συντεταγμένες επάνω.',
    'Undo point': 'Αναίρεση σημείου', 'Cancel': 'Ακύρωση', 'Finish': 'Ολοκλήρωση',
    'Burned area: click points around the perimeter, then Finish.': 'Καμένη έκταση: επιλέξτε σημεία γύρω από την περίμετρο και μετά Ολοκλήρωση.',
    'Firebreak: click points along the line, then Finish.': 'Αντιπυρική ζώνη: επιλέξτε σημεία κατά μήκος της γραμμής και μετά Ολοκλήρωση.',
    'Air drop: click the start and end of the drop line.': 'Ρίψη νερού: επιλέξτε την αρχή και το τέλος της γραμμής ρίψης.',
    'Engine: click the engine position.': 'Όχημα: επιλέξτε τη θέση του οχήματος.',
    'Protected asset: click its location.': 'Προστατευόμενη θέση: επιλέξτε τη θέση της.',
    'Time since ignition': 'Χρόνος από την έναρξη', 'Just now': 'Μόλις τώρα',
    '30 min': '30 λεπτά', '1 h': '1 ώρα', '2 h': '2 ώρες', '3 h': '3 ώρες', '6 h': '6 ώρες', '12 h': '12 ώρες', '24 h': '24 ώρες',
    'Other:': 'Άλλο:', 'hours (0.5 to 48)': 'ώρες (0.5 έως 48)', 'Run forecast': 'Εκτέλεση πρόγνωσης',
    'Direction (wind from)': 'Διεύθυνση (άνεμος από)', 'Speed:': 'Ταχύτητα:', 'Temperature (°C)': 'Θερμοκρασία (°C)',
    'Relative humidity (%)': 'Σχετική υγρασία (%)', 'optional': 'προαιρετικό', 'Apply measurement': 'Εφαρμογή μέτρησης',
    'Name': 'Όνομα', 'e.g. Farmhouse, camp site, school': 'π.χ. αγροικία, κατασκήνωση, σχολείο', 'Add asset': 'Προσθήκη θέσης',
    'Protected asset': 'Προστατευόμενη θέση',
    // compass short (dialog and table)
    'N': 'Β', 'NE': 'ΒΑ', 'E': 'Α', 'SE': 'ΝΑ', 'S': 'Ν', 'SW': 'ΝΔ', 'W': 'Δ', 'NW': 'ΒΔ',
    'NNE': 'ΒΒΑ', 'ENE': 'ΑΒΑ', 'ESE': 'ΑΝΑ', 'SSE': 'ΝΝΑ', 'SSW': 'ΝΝΔ', 'WSW': 'ΔΝΔ', 'WNW': 'ΔΒΔ', 'NNW': 'ΒΒΔ',
    // popups
    'Expected arrival {t}': 'Αναμενόμενη άφιξη {t}', 'In {d}. Earliest {t}': 'Σε {d}. Νωρίτερα {t}',
    'Unlikely': 'Απίθανο', 'Not reached': 'Δεν φτάνει', 'within the forecast window': 'μέσα στο διάστημα πρόγνωσης',
    'Probability of fire: {p}%': 'Πιθανότητα φωτιάς: {p}%', 'Danger: {c}': 'Κίνδυνος: {c}',
    'Potential spread {v} km/h towards the {dir}': 'Δυνητική εξάπλωση {v} km/h προς τα {dir}', ', flames ~{f} m': ', φλόγες ~{f} m',
    'Inside / at the edge of {p}.': 'Εντός / στο όριο: {p}.', 'Would reach {p} in ~{d}.': 'Θα έφτανε στο {p} σε ~{d}.',
    'No village within 6 h of fire travel.': 'Κανένας οικισμός σε απόσταση 6 ωρών εξάπλωσης.',
    '{fuel}, slope {s}%, elevation {e} m': '{fuel}, κλίση {s}%, υψόμετρο {e} m', ', flame length ~{f} m': ', μήκος φλόγας ~{f} m',
    'Add as protected asset': 'Προσθήκη ως προστατευόμενη θέση', 'Outside the forecast area.': 'Εκτός περιοχής πρόγνωσης.',
    'BURNING': 'ΚΑΙΓΕΤΑΙ', 'not reached': 'δεν φτάνει', 'OWN': 'ΔΙΚΗ',
    // busy / toasts
    'Working': 'Επεξεργασία', 'Loading weather, terrain and vegetation': 'Λήψη καιρού, ανάγλυφου και βλάστησης',
    'Updating forecast with current weather': 'Ενημέρωση πρόγνωσης με τρέχοντα καιρό',
    'Forecasting the next {h} h': 'Πρόγνωση για τις επόμενες {h} ώρες',
    'Calibrating the model to the observed fire': 'Βαθμονόμηση του μοντέλου στην παρατηρούμενη φωτιά',
    'Simulating the action': 'Προσομοίωση της ενέργειας', 'Applying measured wind': 'Εφαρμογή μετρημένου ανέμου',
    'Calculating time to impact': 'Υπολογισμός χρόνου άφιξης', 'Re-planning for available resources': 'Επανασχεδιασμός για τα διαθέσιμα μέσα',
    'Running {n} scenarios': 'Εκτέλεση {n} σεναρίων', 'Re-running with wide uncertainty': 'Επανυπολογισμός με ευρεία αβεβαιότητα',
    'Applying action': 'Εφαρμογή ενέργειας', 'Removing action': 'Αφαίρεση ενέργειας',
    'Measured wind applied.': 'Ο μετρημένος άνεμος εφαρμόστηκε.', 'Coordinates copied.': 'Οι συντεταγμένες αντιγράφηκαν.',
    'A burned area needs at least 3 points.': 'Η καμένη έκταση χρειάζεται τουλάχιστον 3 σημεία.',
    'A firebreak needs at least 2 points.': 'Η αντιπυρική ζώνη χρειάζεται τουλάχιστον 2 σημεία.',
    'Enter the forecast window in hours (0.5 to 48).': 'Εισαγάγετε τον ορίζοντα πρόγνωσης σε ώρες (0.5 έως 48).',
    'Location not found. Enter coordinates, e.g. 38.56, 21.94': 'Η θέση δεν βρέθηκε. Εισαγάγετε συντεταγμένες, π.χ. 38.56, 21.94',
    'Close the current incident and start a new one? The current forecast will be cleared.': 'Κλείσιμο του τρέχοντος συμβάντος και έναρξη νέου; Η τρέχουσα πρόγνωση θα διαγραφεί.',
    'now': 'τώρα', 'Language / Γλώσσα': 'Γλώσσα / Language',
    'Server did not answer.': 'Ο διακομιστής δεν απάντησε.', 'Display error: {m}': 'Σφάλμα εμφάνισης: {m}',
  };

  // ── Directions: adverb ("towards the north"), feminine and neuter adjective ──
  const DIRS = {
    'north':      ['βόρεια', 'βόρεια', 'βόρειο'],
    'north-east': ['βορειοανατολικά', 'βορειοανατολική', 'βορειοανατολικό'],
    'east':       ['ανατολικά', 'ανατολική', 'ανατολικό'],
    'south-east': ['νοτιοανατολικά', 'νοτιοανατολική', 'νοτιοανατολικό'],
    'south':      ['νότια', 'νότια', 'νότιο'],
    'south-west': ['νοτιοδυτικά', 'νοτιοδυτική', 'νοτιοδυτικό'],
    'west':       ['δυτικά', 'δυτική', 'δυτικό'],
    'north-west': ['βορειοδυτικά', 'βορειοδυτική', 'βορειοδυτικό'],
  };
  const DIR_RE = '(north-east|north-west|south-east|south-west|north|south|east|west)';
  const dir = (d, form = 0) => (lang === 'el' && DIRS[d] ? DIRS[d][form] : d);

  const KINDS = {
    city: 'πόλη', town: 'κωμόπολη', village: 'χωριό', hamlet: 'οικισμός', suburb: 'συνοικία', hospital: 'νοσοκομείο',
    school: 'σχολείο', kindergarten: 'νηπιαγωγείο', nursing_home: 'γηροκομείο', 'care home': 'γηροκομείο', clinic: 'κλινική',
    camp_site: 'κατασκήνωση', fire_station: 'πυροσβεστικός σταθμός', police: 'αστυνομία', isolated_dwelling: 'μεμονωμένη κατοικία',
    locality: 'τοποθεσία', neighbourhood: 'γειτονιά', farm: 'αγρόκτημα',
  };
  const FUELS = {
    'Pine forest': 'Πευκοδάσος', 'Mountain pine forest': 'Δάσος μαύρης πεύκης', 'Maritime pine forest': 'Δάσος θαλάσσιας πεύκης',
    'Stone pine': 'Κουκουναριά', 'Cypress': 'Κυπαρίσσια', 'Fir forest': 'Ελατοδάσος', 'Oak / broadleaf forest': 'Δρυοδάσος / πλατύφυλλα',
    'Chestnut forest': 'Καστανοδάσος', 'Beech forest': 'Οξιοδάσος', 'Dense shrubland': 'Πυκνοί θαμνώνες', 'Tall shrubland': 'Υψηλοί θαμνώνες',
    'Low scrub': 'Φρύγανα', 'Scrub / garrigue': 'Θαμνώνες', 'Dry grass': 'Ξηρά χόρτα', 'Fields / stubble': 'Αγροί / καλαμιές',
    'Overgrown farmland': 'Εγκαταλελειμμένες καλλιέργειες', 'Eucalyptus': 'Ευκάλυπτοι', 'Olive grove': 'Ελαιώνας', 'Vineyard': 'Αμπελώνας',
    'River vegetation': 'Παρόχθια βλάστηση', 'Water': 'Νερό', 'Built-up area': 'Δομημένη περιοχή', 'Road': 'Δρόμος', 'Bare / rock': 'Γυμνό έδαφος / βράχος',
  };
  const RES = { 'aircraft': 'εναέριο μέσο', 'fire truck': 'όχημα', 'fire trucks': 'οχήματα', 'bulldozer': 'χωματουργικό', 'bulldozers': 'χωματουργικά' };
  const hours = (n) => `${n} ${+n === 1 ? 'ώρα' : 'ώρες'}`;
  const T = '(\\{t:\\d+\\})';
  const Q = (s) => `«${s}»`;

  // ── Engine sentences: [pattern, replacement] applied in order ──
  const RULES = [
    // actions: titles
    [/Warn \/ evacuate (.+)/, (m, p) => `Ειδοποίηση / εκκένωση: ${p}`],
    [/Cut a firebreak across the head of the fire/, 'Διάνοιξη αντιπυρικής ζώνης μπροστά από το μέτωπο'],
    [/Cut a firebreak to protect (.+)/, (m, p) => `Διάνοιξη αντιπυρικής ζώνης για την προστασία: ${p}`],
    [/Aircraft: drops to shield (.+)/, (m, p) => `Εναέρια μέσα: ρίψεις για την προστασία: ${p}`],
    [/Aircraft: drops on the head of the fire/, 'Εναέρια μέσα: ρίψεις στο μέτωπο'],
    [new RegExp(`Aircraft: drops on the ${DIR_RE} front`), (m, d) => `Εναέρια μέσα: ρίψεις στο ${DIRS[d][2]} μέτωπο`],
    [/Fire truck to protect (.+)/, (m, p) => `Όχημα για την προστασία: ${p}`],
    [/Fire truck on the (left|right) flank/, (m, s) => `Όχημα στην ${s === 'left' ? 'αριστερή' : 'δεξιά'} πλευρά`],
    // actions: details
    [new RegExp(`Fire may reach (.+?) around ${T} \\((\\d+)% of forecasts\\)\\. Start warning people now\\.`),
      (m, p, t, n) => `Η φωτιά μπορεί να φτάσει στο ${Q(p)} περίπου στις ${t} (${n}% των προγνώσεων). Ξεκινήστε άμεσα την ειδοποίηση των κατοίκων.`],
    [/([\d.]+ k?m) long, at least ([\d.]+ k?m) wide \(bulldozer \+ crew, ~(\d+) min\)\./,
      (m, l, w, n) => `Μήκος ${l}, πλάτος τουλάχιστον ${w} (χωματουργικό και συνεργείο, ~${n} λεπτά).`],
    [/Both ends tie into ground that does not burn\./, 'Και τα δύο άκρα καταλήγουν σε έδαφος που δεν καίγεται.'],
    [/One end ties into ground that does not burn\./, 'Το ένα άκρο καταλήγει σε έδαφος που δεν καίγεται.'],
    [/Ends are not anchored — watch the flanks\./, 'Τα άκρα δεν είναι αγκυρωμένα. Προσοχή στις πλευρές.'],
    [new RegExp(`Start before ${T}; fire could reach the line from ${T}\\.`),
      (m, a, b) => `Έναρξη πριν τις ${a}. Η φωτιά μπορεί να φτάσει στη γραμμή από τις ${b}.`],
    [/High flames \(([\d.]+ m)\): expect spot fires across the line\./, (m, f) => `Υψηλές φλόγες (${f}): αναμένονται δευτερογενείς εστίες πέρα από τη γραμμή.`],
    [new RegExp(`~400 m drop line across the fire's path, first drop at ${T}, then repeat every ~45 min for (\\d+) h\\.`),
      (m, t, h) => `Γραμμή ρίψης ~400 m εγκάρσια στην πορεία της φωτιάς, πρώτη ρίψη στις ${t}, επανάληψη κάθε ~45 λεπτά για ${hours(h)}.`],
    [/Extreme fire: drops will slow it but not stop it\./, 'Ακραία συμπεριφορά: οι ρίψεις θα την επιβραδύνουν αλλά δεν θα τη σταματήσουν.'],
    [/Defend the edge of (.+?)\. /, (m, p) => `Άμυνα στην παρυφή του ${Q(p)}. `],
    [/Hold position between the fire and (.+?)\. /, (m, p) => `Θέση ανάμεσα στη φωτιά και το ${Q(p)}. `],
    [/Position is on a road\./, 'Η θέση βρίσκεται σε δρόμο.'],
    [/No road found nearby — check access\./, 'Δεν βρέθηκε δρόμος κοντά. Ελέγξτε την πρόσβαση.'],
    [new RegExp(`Fire could arrive from ~${T}\\.`), (m, t) => `Η φωτιά μπορεί να φτάσει από ~${t}.`],
    [/Flames ~([\d.]+ m) — too intense for direct attack: protect buildings and people\./,
      (m, f) => `Φλόγες ~${f}: πολύ έντονη για άμεση προσβολή. Προστασία κτιρίων και ανθρώπων.`],
    [new RegExp(`Attack the ${DIR_RE} flank, working from the burned side\\.`), (m, d) => `Προσβολή στη ${DIRS[d][1]} πλευρά, από την καμένη έκταση.`],
    [/Flames there ~([\d.]+ m)\./, (m, f) => `Φλόγες εκεί ~${f}.`],
    // planner: effects and notes
    [/(\d+) (aircraft|fire trucks?|bulldozers?) not needed in this plan — keep in reserve\./,
      (m, n, r) => `${n} ${r === 'aircraft' && +n > 1 ? 'εναέρια μέσα' : RES[r]}: δεν χρειάζονται σε αυτό το σχέδιο. Διατήρηση σε εφεδρεία.`],
    [/With 1 more (aircraft|fire truck|bulldozer): /, (m, r) => `Με 1 ακόμη ${RES[r]}: `],
    [/(^|[^ε] )1 more (aircraft|fire truck|bulldozer): |^1 more (aircraft|fire truck|bulldozer): /, (m, pre, r1, r2) => `${pre || ''}1 ακόμη ${RES[r1 || r2]}: `],
    [/[Kk]eeps the fire out of (.+?) in (\d+)% of scenarios/, (m, p, n) => `κρατά τη φωτιά μακριά από το ${Q(p)} στο ${n}% των σεναρίων`],
    [/[Ff]ire reaches (.+?) ~(\d+ h(?: \d+ min)?|\d+ min) later/, (m, p, d) => `η φωτιά φτάνει στο ${Q(p)} ~${d} αργότερα`],
    [/[Ss]aves ~(\d+) ha/, (m, n) => `περιορίζει την καμένη έκταση κατά ~${n} ha`],
    [/Little effect\./, 'Μικρή επίδραση.'],
    [/On its own: /, 'Μεμονωμένα: '],
    [/would help further/, 'θα βοηθούσε περαιτέρω'],
    // fire behaviour advice
    [/Low intensity: hand crews can attack directly\./, 'Χαμηλή ένταση: άμεση προσβολή από πεζοπόρα τμήματα.'],
    [/Moderate: engines and hand crews can attack the edge\./, 'Μέτρια ένταση: προσβολή της παρυφής από οχήματα και πεζοπόρα τμήματα.'],
    [/High: use heavy equipment and aircraft at the head\./, 'Υψηλή ένταση: βαριά μέσα και εναέρια στο μέτωπο.'],
    [/Very high: do not attack the head directly — work the flanks, cut breaks ahead, protect people\./,
      'Πολύ υψηλή ένταση: όχι άμεση προσβολή του μετώπου. Εργασία στις πλευρές, αντιπυρικές ζώνες μπροστά, προστασία ανθρώπων.'],
    [new RegExp(`Wind turns to blow from the ${DIR_RE} around ${T} — the fire will change direction\\.`),
      (m, d, t) => `Ο άνεμος στρέφεται και θα πνέει από ${DIRS[d][0]} περίπου στις ${t}. Η φωτιά θα αλλάξει κατεύθυνση.`],
    // confidence reasons
    [/Wind measured on site\./, 'Άνεμος μετρημένος επί τόπου.'],
    [/No weather forecast — report the local wind\./, 'Χωρίς μετεωρολογική πρόγνωση. Καταχωρίστε τον τοπικό άνεμο.'],
    [/Wind from the weather forecast \(not measured on site\)\./, 'Άνεμος από τη μετεωρολογική πρόγνωση (όχι μετρημένος επί τόπου).'],
    [/Vegetation map missing — generic fuel assumed\./, 'Λείπει ο χάρτης βλάστησης: χρησιμοποιήθηκε γενική καύσιμη ύλη.'],
    [/Terrain missing — flat ground assumed\./, 'Λείπει το ανάγλυφο: θεωρήθηκε επίπεδο έδαφος.'],
    [/Adjusted to the burned area seen on the ground \((\d+)% match(?:, fire edge within ~(\d+) m)?\)\./,
      (m, p, e) => `Προσαρμοσμένο στην καμένη έκταση που παρατηρήθηκε (${p}% συμφωνία${e ? `, όριο φωτιάς εντός ~${e} m` : ''}).`],
    [/Last check: the forecast made (.+?) before the latest observation (?:put the fire edge within ~(\d+) m of it|matched it) \((\d+)% match\)\./,
      (m, d, e, p) => `Τελευταίος έλεγχος: η πρόγνωση που έγινε ${d} πριν από την τελευταία παρατήρηση ${e ? `είχε απόκλιση ~${e} m στο όριο της φωτιάς` : 'συμφώνησε'} (${p}% συμφωνία).`],
    [/Not yet checked against the real fire — mark the burned area to improve accuracy\./,
      'Δεν έχει ελεγχθεί ακόμη με την πραγματική φωτιά. Σημειώστε την καμένη έκταση για μεγαλύτερη ακρίβεια.'],
    [/Forecasts disagree a lot on how far the fire gets\./, 'Τα σενάρια διαφέρουν πολύ ως προς το πόσο μακριά φτάνει η φωτιά.'],
    [/Wind change expected during the forecast\./, 'Αναμένεται αλλαγή ανέμου μέσα στο διάστημα πρόγνωσης.'],
    // calibration
    [/Model matched to the observed fire: (\d+)% → (\d+)% agreement on the new growth/,
      (m, a, b) => `Το μοντέλο προσαρμόστηκε στην παρατηρούμενη φωτιά: συμφωνία στη νέα εξάπλωση ${a}% → ${b}%`],
    [/, fire edge within ~(\d+) m/, (m, e) => `, όριο φωτιάς εντός ~${e} m`],
    [/Near the front the spread is corrected ×([\d.]+)–×([\d.]+) by direction/,
      (m, a, b) => `Κοντά στο μέτωπο η ταχύτητα εξάπλωσης διορθώνεται ×${a}–×${b} ανά κατεύθυνση`],
    // warnings
    [/Terrain unavailable — assuming flat ground\./, 'Το ανάγλυφο δεν είναι διαθέσιμο: θεωρείται επίπεδο έδαφος.'],
    [/Land-cover map unavailable — using generic Mediterranean scrub everywhere\./,
      'Ο χάρτης κάλυψης γης δεν είναι διαθέσιμος: χρησιμοποιείται γενική μεσογειακή θαμνώδης βλάστηση.'],
    [/No weather forecast available — using typical summer conditions .*?Report the local wind to improve the forecast\./,
      'Δεν υπάρχει μετεωρολογική πρόγνωση: χρησιμοποιούνται τυπικές θερινές συνθήκες (30 °C, 25 % υγρασία, βόρειος άνεμος 18 km/h). Καταχωρίστε τον τοπικό άνεμο για καλύτερη πρόγνωση.'],
    [/Villages and roads could not be loaded .*?by hand\./,
      'Οι οικισμοί και οι δρόμοι δεν φορτώθηκαν (το OpenStreetMap δεν είναι διαθέσιμο). Προσθέστε θέσεις χειροκίνητα με την Προστασία θέσης.'],
    // server status and errors
    [/Choosing the best plan for your resources…/, 'Επιλογή του καλύτερου σχεδίου για τα διαθέσιμα μέσα…'],
    [/Computing forecast…/, 'Υπολογισμός πρόγνωσης…'],
    [/Fire may leave the map — enlarging the forecast area…/, 'Η φωτιά μπορεί να βγει από τον χάρτη: διεύρυνση της περιοχής πρόγνωσης…'],
    [/Getting weather…/, 'Λήψη καιρού…'],
    [/Loading terrain and vegetation maps…/, 'Φόρτωση ανάγλυφου και βλάστησης…'],
    [/Matching the model to the real fire…/, 'Βαθμονόμηση του μοντέλου στην πραγματική φωτιά…'],
    [/Running (\d+) scenarios…/, (m, n) => `Εκτέλεση ${n} σεναρίων…`],
    [/Updating forecast…/, 'Ενημέρωση πρόγνωσης…'],
    [/No vegetation that can burn at this location \(water, rock or built-up\)\./, 'Δεν υπάρχει καύσιμη βλάστηση σε αυτή τη θέση (νερό, βράχος ή δομημένη περιοχή).'],
    [/No active fire.*/, 'Δεν υπάρχει ενεργό συμβάν.'],
    [/The burned area is outside the map area\./, 'Η καμένη έκταση είναι εκτός χάρτη.'],
    [/That place is outside the forecast area\./, 'Η θέση είναι εκτός περιοχής πρόγνωσης.'],
    [/Observation is older than the latest known fire state\./, 'Η παρατήρηση είναι παλαιότερη από την τελευταία γνωστή κατάσταση της φωτιάς.'],
    [/Coordinates out of range\./, 'Συντεταγμένες εκτός ορίων.'],
    [/fire kept out/, 'η φωτιά δεν φτάνει'],
    [/^\+(.+) later$/, (m, d) => `+${d} αργότερα`],
    [/^Open-Meteo forecast$/, 'Πρόγνωση Open-Meteo'],
    // data sources
    [/^measured on site$/, 'μετρημένος επί τόπου'],
    [/^default summer conditions \(no forecast\)$/, 'τυπικές θερινές συνθήκες (χωρίς πρόγνωση)'],
    // durations left in text ("1 h 30 min", "45 min", "+2 h later")
    [/\b(\d+) h (\d+) min\b/g, (m, h, n) => `${hours(h)} ${n} λεπτά`],
    [/\b(\d+) min\b/g, (m, n) => `${n} λεπτά`],
    [/\b(\d+) h\b(?!\w)/g, (m, n) => hours(n)],
  ];

  function fill(s, vars) {
    return vars ? s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m)) : s;
  }

  return {
    get lang() { return lang; },
    set(l) {
      lang = l === 'el' ? 'el' : 'en';
      try { localStorage.setItem('wilson-lang', lang); } catch (_) {}
      document.documentElement.lang = lang;
    },
    t(s, vars) { return fill(lang === 'el' && s in EL ? EL[s] : s, vars); },
    // Engine sentence(s) → current language.
    tr(s) {
      if (lang !== 'el' || !s) return s || '';
      let out = String(s);
      for (const [re, rep] of RULES) out = out.replace(re, rep);
      return out;
    },
    dir,
    danger: (i) => (lang === 'el' ? ['Κανένας', 'Χαμηλός', 'Μέτριος', 'Υψηλός', 'Πολύ υψηλός', 'Ακραίος']
      : ['None', 'Low', 'Moderate', 'High', 'Very high', 'Extreme'])[i] || '',
    kind: (k) => (lang === 'el' && KINDS[k]) || k, fuel: (f) => (lang === 'el' && FUELS[f]) || f,
    // Translates static markup: data-i18n (text), data-i18n-title, data-i18n-ph (placeholder).
    apply(root = document) {
      for (const el of root.querySelectorAll('[data-i18n]')) {
        if (!el.dataset.en) el.dataset.en = el.textContent.trim();
        el.textContent = this.t(el.dataset.en);
      }
      for (const el of root.querySelectorAll('[data-i18n-title]')) {
        if (!el.dataset.enTitle) el.dataset.enTitle = el.title;
        el.title = this.t(el.dataset.enTitle);
      }
      for (const el of root.querySelectorAll('[data-i18n-ph]')) {
        if (!el.dataset.enPh) el.dataset.enPh = el.placeholder;
        el.placeholder = this.t(el.dataset.enPh);
      }
      document.documentElement.lang = lang;
    },
  };
})();
const t = (s, vars) => I18N.t(s, vars);
