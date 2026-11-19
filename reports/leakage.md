# Near-Duplicate Leakage Report

For every TEST message we look for the most similar TRAIN message.

- **exact**: same text after normalizing (lowercase, links -> `<url>`, digits -> `0`, no punctuation, single spaces).
- **near**: char TF-IDF cosine similarity >= 0.9 (includes exact).
- **old split**: the random stratified split used up to Step 6 (rebuilt here with the same seed).
- **clean split**: Step 7a split by near-duplicate cluster (a cluster is never in both train and test).

## Summary

| Language | Split | Test rows | Exact dupes | Exact % | Near dupes | Near % | Near % (spam) | Near % (ham) |
|---|---|---|---|---|---|---|---|---|
| en | old | 1032 | 16 | 1.6 | 53 | 5.1 | 33.6 | 1.1 |
| en | clean | 1032 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 |
| tr | old | 948 | 37 | 3.9 | 108 | 11.4 | 19.4 | 2.3 |
| tr | clean | 948 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 |

## Examples: en, old split (5 random leaked pairs)

| Similarity | Exact | Test message | Closest train message |
|---|---|---|---|
| 1.000 | yes | Send a logo 2 ur lover - 2 names joined by a heart. Txt LOVE NAME1 NAME2 MOBNO eg LOVE ADAM EVE 07123456789... | Send a logo 2 ur lover - 2 names joined by a heart. Txt LOVE NAME1 NAME2 MOBNO eg LOVE ADAM EVE 07123456789... |
| 0.943 | no | URGENT! Your mobile number *************** WON a £2000 Bonus Caller prize on 10/06/03! This is the 2nd atte... | URGENT! Your mobile No *********** WON a £2,000 Bonus Caller Prize on 02/06/03! This is the 2nd attempt to ... |
| 0.987 | no | ree entry in 2 a weekly comp for a chance to win an ipod. Txt POD to 80182 to get entry (std txt rate) T&C'... | Free entry in 2 a weekly comp for a chance to win an ipod. Txt POD to 80182 to get entry (std txt rate) T&C... |
| 1.000 | yes | Customer service announcement. We recently tried to make a delivery to you but were unable to do so, please... | Customer service announcement. We recently tried to make a delivery to you but were unable to do so, please... |
| 0.949 | no | URGENT! Your Mobile number has been awarded a <UKP>2000 prize GUARANTEED. Call 09061790125 from landline. C... | URGENT! Your Mobile number has been awarded with a £2000 prize GUARANTEED. Call 09061790121 from land line.... |

## Examples: tr, old split (5 random leaked pairs)

| Similarity | Exact | Test message | Closest train message |
|---|---|---|---|
| 0.995 | no | Garanti ile calismaya devam edin kredi dunyasinin ayricaliklarini kacirmayin! Size ozel kampanyalarimizla i... | Garanti ile calismaya devam edin kredi dunyasinin ayricaliklarini kacirmayin! Size ozel kampanyalarimizla i... |
| 1.000 | yes | Denize Sifir Oasis Eternity Alanya Hotel de Kampanya! 4 Gece 5 Gun Hersey Dahil Tatil 359 TL Hemen Ara 0242... | Denize Sifir Oasis Eternity Alanya Hotel de Kampanya! 4 Gece 5 Gun Hersey Dahil Tatil 279 TL Hemen Ara 0242... |
| 0.958 | no | Gunun yorgunlugunu bizimle atmak istermisiniz?Memnuniyet garantili yerimiz BAKIRKOY Incirlide 10:00-02:00 a... | Gunun Yorgunlugunu Bizimle Atmak Istermisiniz? Memnuniyet Garantili Yerimiz BAKIRKOY Incirlide 10:00-02:00 ... |
| 0.915 | no | KAMPANYA: LOFT'TA TUM URUNLERDE ETIKETIN YARISI INDIRIMI BASLAMISTIR. SMS ALMAMAK ICIN LOFTSU YAZIP 7889'A ... | KAMPANYA: KISIN SON INDIRIMI LOFT'TA TUM URUNLER ETIKETIN YARISI. SMS ALMAMAK ICIN LOFTSU YAZIP 7889'A UCRE... |
| 0.990 | no | GARANTI'den Kis Kredisi firsati! Ihtiyaclarinizi ertelemeyin, cok avantajli destek kredileri icin en yakin ... | GARANTI'den Kis Kredisi firsati! Ihtiyaclarinizi ertelemeyin, cok avantajli destek kredileri icin en yakin ... |
