# Turkish Phishing Challenge Set

- Model: champion (registered version 13, models/model.joblib)
- Messages: 40 (hand-written; evaluation only, never trained on)

| Metric | Value |
|---|---|
| challenge_accuracy | 0.9250 |
| challenge_precision | 1.0000 |
| challenge_recall | 0.8500 |
| challenge_f1 | 0.9189 |

## Misclassified messages (3)

- true **spam** -> predicted ham = phishing we MISSED
- true **ham** -> predicted spam = normal SMS we would block

| True label | Spam prob | Message |
|---|---|---|
| spam | 0.363 | Hesabınıza yeni bir cihazdan giriş yapıldı. Siz değilseniz şifrenizi buradan sıfırlayın: https://guvenli-giris.example.com |
| spam | 0.415 | Operatörünüz: Yeni SIM kartınız için bilgilerinizi doğrulayın, aksi halde hattınız askıya alınacak. example.com/sim |
| spam | 0.286 | Merhaba anne, telefonum bozuldu bu yeni numaram. Acil 2000 TL lazım bu IBAN'a atar mısın TR00 0000 0000 0000 0000 0000 00 |
