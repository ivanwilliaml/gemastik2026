# title: Hitung bahasa daerah: 10 bahasa dengan kata penanda
# when: EDA, cukup detik. Menentukan perlu tidaknya model multibahasa. Di atas sekitar 5% (aturan praktis), coba NusaBERT.
# avoid: Menganggap hasilnya presisi: komentar pendek dan campur kode sering lolos. Angka ini batas kasar, baca sampelnya.
# tags: bahasa daerah jawa sunda minang bali banjar madura aceh bugis batak ngaju code-mixing campur
import re
from collections import Counter
import numpy as np
import pandas as pd

# Kata yang sering di NusaX tiap bahasa dan jarang di korpus Indonesia acuan. Teruji pada NusaX test.
LANG_MARKERS = {
    "jav": "lan ing ning kanggo mangan uga nanging akeh panganane panganan dadi saka panggonan mung amarga kene rasane regane wis dhewe rega pelayanane marang uwis panggonane lak sek kuwi utawa panggon maneh luwih kanthi dinggo nganggo yaiku wektu uwong nggawe merga mbengi saiki panggone liyane kutha teka dina ngenteni kaluwarga arep adoh nganti mlebu dingge grapyak dalan awakdewe kanca entuk tuku siji nggarai mesthi mesen iwak suwi sega pithik mrene wengi kowe elek awan resik pemandangane omah njajal meh ben pilian ora ndak ndk mboten nggih ngono ngene awakmu piye durung wes karo ndang pancen ojo sampun panjenengan matur nuwun kulo sopo opo koyo enek duwit sego dewe podo seng lek kon ae wae",
    "sun": "sareng teu abdi ieu kanggo oge tiasa tuangeun tuang mung jeung seueur dieu sae dina tos kusabab tempatna upami rasana pikeun eta kuring tuangeunna hoyong deui cekap henteu arurang nyaeta cocog awis sumping wengi pangaos merenah mesen ayeuna sanes janten urang sawaktos rupi-rupi pangaosna sababaraha dinten anjeun teuing sapertos palayanan rerencangan pamandangan kulawargi kitu atawa ngeunah nganggo hayam bade keneh nepi leuwih keur mirah hungkul enggal mimiti kirang disarankeun dugi disayogikeun waktos tina ningali nguciwakeun lebet kedah jalmi tabuh sahiji atuh euy naon kumaha geus moal meni hatur nuhun kieu tong loba siga kunaon",
    "min": "untuak lamak indak bana ado tampek wak siko sangaik tampeknyo raso nio makanannyo harago cukuik samo rasonyo urang haragonyo karano juo labiah iko rancak pasan hanyo suko pamandangan biaso palayanan mambuek kaluarga dapek tibo piliahan paliang lua jauah buek ancak sadang masuak taruih sasuai sahinggo gadang baragam wakatu punyo manunggu lamo tajangkau sinan sajuak karajo disajian baliak salain patang mamasan kumuah talampau sanak sambia manuruik liak cubo basamo bantuak babarapo rami pai manikmati lueh apolai naiak mancari goreang direkomendasian dakek",
    "ban": "lan sane sajan ajengan ngajeng tiyang jaen nanging krana ten liu jaan tongos wenten tusing lakar utawi puniki sampun dados yening iraga anggon genah ajine ene uli luung dadi punika gen rasane nenten mael becik asane ajenganne tongosne sareng kanti buin ajengane niki suba dugas mesen ento antuk ngaenang lianan indik akeh driki dot peteng timpal ngranayang genahne jani ragane pengarga dina sekadi asapunika ngae anggen dogen teka nganggen kuciwa tiange taen meli makejang kuang bes patuh ngalih kasajiang enggal",
    "bjn": "kada gasan ulun banar jua kawa haja wadahnya handak ngini gin tagal haraga pang urang maulah haraganya risturan amun liwar katuju mayan palayanan ngitu cucuk ristu buhan pamandangan mun wayah pina kaluarga matan atawa mambari hawanya wahini sualnya kalu sasuai pian palayanannya labih hanyar babaya tajangkau rami parak hudah bamacam sakalinya imbah talalu mamasan lamahan babarapa tarus suah kadida kadada hancap sangkal pinanya muar malihat mahadang sigar sadang sabarataan rigat intang ikam disuruhakan bujur bubuhan manikmati katujuan jar gonol disurungakan",
    "mad": "bik ben engkok ghebey sarah kakanan ngakan riya tape rua mun bede kennengngan polana kiya kakananna rassana deri oreng deddhi terro pole cokop rassa bhei bhegus ongghu bekto diye regghena ghabay bennyak engko' deteng bennya' regghe olle pera' otabe mude lebur bekna senneng lebbi kennengnganna paleng ghuring selaen satiya pelean tadek kodhu kanca jhet settong pade kotta ta' messen korang kia entar bhuru aghebey pelayananna jhek ghik delem masok mare lerres acem-macem kenengngan ajem abit sampe' pendhenan padena lekkas bada",
    "ace": "hana ngon mangat lon nyoe pih keu nyan cit meunyo jeuet jih jak lagee watee pajoh mantong teumpat pah lom cukop kamoe ureueng get meuhai rumoh lam uroe saboh leubeh jeut teumpatjih lagak disinoe makananjih boh peunajoh atawa lheueh kon keuneuk rasajih loen leupah saweueb peue paleng ngen trok lee keunoe keuh seubab peulayanan droe sep sagai peuget keuluarga jinoe manok duek sabe phon ngoen inoe 'oh yumjih trep rakan preh peugah baro teungoh bagah tapajoh nakeuh lheuh gureng aneuk",
    "bug": "okko sibawa lao ladde' malunra' manre engka anre-anre ellinna aga pole rasana nasaba yako makanja' maneng maega okkoe anroang de' anroanna nasaba' iya' makanja iyaro restorang topa nappa wettu rupanna narekko makessing wenni onroang ladde manyameng anre mancaji makae kanja' iyye melo onroanna iyae pakkanreanna masempo kaminang makkokkoe masoli lunra' massuroang de'na wette malunra makkada maelo jaji tette' pajjamana okkoro laleng yaregga pirasai pakkanreang manu' mancajiwi iyanaritu esso deggaga sedding laingnge cedde' wettunna tennia taue sikenangeng runtu' meloki lettu' lettu",
    "bbc": "dohot dang tabo sipanganon jala hian adong inganan alai sian mansai gabe laho godang songon boi mangan naeng argana muse roha holan arga tung mambahen alana daina sahat manang dison parmanganan ikkon ingananna denggan hami akka lomo ima ahu tingki dope ndang borngin jolma dai bagak alani sipanganonna manuhor unang nai mura dalan sipilliton nungga hatop halak dongan uli sude boe panghobasionna lao panatapan marragam somal rap parjolo lambok tuson torus hurang sada iba dao sonang piga-piga menak lam panghobasion",
    "nij": "mangat eka tutu penginan diak kuman hong tege angat bahalap tuntang kea awi baya handak rega mawi hetuh hung ije uluh jituh beken amun panginan huang kilau ikei bujur andau dumah due puna hayak hindai wayah gin tame sukup hamalem lalau ewen labih meteh jatunti uras tempayah lime puluh mun haru sadang kawal hapan dinun manuk mengkeme ela kanih jatun intihan capat ampi telu munduk tukep nyerungan itah ikau belai sanang isut sinde pekasak mihup kurik dahang yuh ulih mite"
}
LANG_NAMES = {"ind": "Indonesia/lainnya", "jav": "Jawa", "sun": "Sunda", "min": "Minang", "ban": "Bali", "bjn": "Banjar", "mad": "Madura", "ace": "Aceh", "bug": "Bugis", "bbc": "Batak Toba", "nij": "Ngaju"}
_MARK = {k: set(v.split()) for k, v in LANG_MARKERS.items()}

def detect_local_language(text, min_hits=2):
    """min_hits=2 menjaga presisi. min_hits=1 untuk batas atas: lebih banyak salah."""
    w = set(re.findall(r"[a-zà-ÿ'’-]{3,}", str(text).lower()))
    best, hits = "ind", 0
    for code, m in _MARK.items():
        h = len(w & m)
        if h > hits:
            best, hits = code, h
    return best if hits >= min_hits else "ind"

def language_report(texts, show=5, seed=42):
    """Porsi tiap bahasa pada teks unik, batas ketat dan batas longgar, plus contoh untuk dibaca."""
    u = pd.Series(list(texts)).dropna().astype(str).drop_duplicates()
    strict, loose = u.map(detect_local_language), u.map(lambda t: detect_local_language(t, 1))
    tbl = pd.DataFrame({"ketat": strict.value_counts(), "longgar": loose.value_counts()}).fillna(0).astype(int)
    tbl.index = [LANG_NAMES.get(i, i) for i in tbl.index]
    tbl["ketat_%"] = (100 * tbl.ketat / len(u)).round(2)
    tbl["longgar_%"] = (100 * tbl.longgar / len(u)).round(2)
    local = float((strict != "ind").mean())
    upper = float((loose != "ind").mean())
    print(tbl.sort_values("longgar", ascending=False).to_string())
    print(f"Bahasa daerah: {local:.2%} (ketat) sampai {upper:.2%} (longgar) dari {len(u)} teks unik.")
    print("Saran:", "coba NusaBERT dan evaluasi per bahasa." if local >= .05 else "pantau dengan batas longgar; bila di atas 5%, baca sampel lalu coba NusaBERT." if upper >= .05 else "tidak perlu alur khusus; char n-gram cukup.")
    hit = u[strict != "ind"]
    for t in hit.sample(min(show, len(hit)), random_state=seed):
        print("  -", t[:100].replace("\n", " "))
    return strict.reindex(u.index)
