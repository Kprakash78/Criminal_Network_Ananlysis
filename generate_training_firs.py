import os

cases = [
    ("FIR_038", "Sheena Bora", "FIR_2026_00038", "25/08/2015", "Mumbai", "Indrani Mukerjea, Sanjeev Khanna, Shyamvar Pinturam Rai", "Sheena Bora", "Murder and disposal of body in Raigad forest. Vehicle MH04FZ3421 used."),
    ("FIR_039", "Satyam Scam", "FIR_2026_00039", "09/01/2009", "Hyderabad", "Ramalinga Raju, Rama Raju, Vadlamani Srinivas", "Satyam Computer Services Shareholders", "Corporate fraud of 7136 Crores. Falsification of bank accounts including ACC89012 and ACC99123."),
    ("FIR_040", "Telgi Scam", "FIR_2026_00040", "19/08/2001", "Pune", "Abdul Karim Telgi", "Government of India", "Counterfeit stamp paper scam worth over 30000 Crores. Multiple printing presses across state borders."),
    ("FIR_041", "2G Spectrum", "FIR_2026_00041", "21/10/2009", "Delhi", "A. Raja, Siddharth Behura, R. K. Chandolia", "Government of India", "Telecom spectrum allocation fraud causing massive exchequer loss."),
    ("FIR_042", "Nithari Killings", "FIR_2026_00042", "29/12/2006", "Noida", "Moninder Singh Pandher, Surinder Koli", "Multiple children", "Serial killings and abduction in Sector 31 Noida."),
    ("FIR_043", "Aarushi Talwar", "FIR_2026_00043", "16/05/2008", "Noida", "Rajesh Talwar, Nupur Talwar (Initial Suspects)", "Aarushi Talwar, Hemraj", "Double murder at L-32 Jalvayu Vihar. High profile case with botched initial investigation."),
    ("FIR_044", "Chhota Rajan Extortion", "FIR_2026_00044", "12/04/2003", "Mumbai", "Rajendra Sadashiv Nikalje", "Builders Association", "Extortion calls made from phone 9876541111 demanding protection money."),
    ("FIR_045", "Badaun Hangings", "FIR_2026_00045", "28/05/2014", "Badaun", "Pappu Yadav, Awadhesh Yadav, Urvesh Yadav", "Two minor cousins", "Abduction and murder resulting in bodies hung from a mango tree."),
    ("FIR_046", "Vyapam Scam", "FIR_2026_00046", "07/07/2013", "Bhopal", "Laxmikant Sharma, Pankaj Trivedi", "MPPEB", "Massive admission and recruitment scam in Madhya Pradesh using proxies."),
    ("FIR_047", "Coal Scam", "FIR_2026_00047", "03/09/2012", "Delhi", "Unknown Government Officials, Corporate Entities", "Government of India", "Irregularities in allocation of coal blocks causing massive exchequer loss."),
    ("FIR_048", "Fodder Scam", "FIR_2026_00048", "27/01/1996", "Patna", "Lalu Prasad Yadav, Jagannath Mishra", "Animal Husbandry Department", "Embezzlement of 940 Crores from state treasury through fake bills."),
    ("FIR_049", "Harshad Mehta", "FIR_2026_00049", "23/04/1992", "Mumbai", "Harshad Mehta, Ashwin Mehta", "SBI", "Stock market manipulation using fake bank receipts. Account ACC11223 used for siphoning."),
    ("FIR_050", "Ketan Parekh", "FIR_2026_00050", "30/03/2001", "Mumbai", "Ketan Parekh", "Bank of India", "Circular trading and market manipulation using pay orders from Madhavpura Mercantile Co-operative Bank."),
    ("FIR_051", "Saradha Chit Fund", "FIR_2026_00051", "15/04/2013", "Kolkata", "Sudipto Sen, Debjani Mukherjee", "Investors", "Ponzi scheme defrauding over 1.7 million investors of nearly 200 to 300 billion rupees."),
    ("FIR_052", "Rose Valley Scam", "FIR_2026_00052", "05/03/2014", "Kolkata", "Gautam Kundu", "Investors", "Massive financial fraud larger than Saradha, collecting over 15000 Crores illegally."),
    ("FIR_053", "Nirav Modi PNB", "FIR_2026_00053", "29/01/2018", "Mumbai", "Nirav Modi, Mehul Choksi", "Punjab National Bank", "Fraudulent Letters of Undertaking (LoUs) causing 14000 Crore loss to PNB."),
    ("FIR_054", "Vijay Mallya IDBI", "FIR_2026_00054", "10/10/2015", "Mumbai", "Vijay Mallya", "IDBI Bank", "Wilful default and money laundering of loans worth 9000 Crores via Kingfisher Airlines."),
    ("FIR_055", "AgustaWestland", "FIR_2026_00055", "12/03/2013", "Delhi", "S.P. Tyagi, Christian Michel", "Ministry of Defence", "Bribery scandal in the procurement of 12 VVIP helicopters from Finmeccanica."),
    ("FIR_056", "Bofors Scandal", "FIR_2026_00056", "22/01/1990", "Delhi", "Ottavio Quattrocchi, Win Chadha", "Government of India", "Kickbacks paid in the 1.4 billion dollar deal between India and Swedish arms manufacturer AB Bofors."),
    ("FIR_057", "Hawala Scandal", "FIR_2026_00057", "03/03/1991", "Delhi", "Surendra Kumar Jain", "Multiple Politicians", "Bribery using hawala channels for funding Kashmiri militants and paying politicians. Account ACC44556 linked."),
    ("FIR_058", "Kidnapping Template", "FIR_2026_00058", "10/10/2026", "Academy", "Ramesh Singh", "Sunil Verma", "Training case: Victim kidnapped in vehicle DL9CX1234. Ransom calls made from phone 9876545555."),
    ("FIR_059", "Chain Snatching Template", "FIR_2026_00059", "11/10/2026", "Academy", "Unknown Bike Riders", "Geeta Sharma", "Training case: Two men on a black Pulsar snatched gold chain. Fled towards MG Road."),
    ("FIR_060", "Organized Crime Template", "FIR_2026_00060", "12/10/2026", "Academy", "D-Company Associates", "Local Businesses", "Training case: Protection racket run by associates of Dawood Ibrahim. Extortion money sent to account ACC88990.")
]

template = """FIRST INFORMATION REPORT
FIR No.: {fir_no}
Date: {date}
Police Station: {station}
Reporting Officer: Training Academy Staff
Case Reference: {ref}

INCIDENT REPORT:
{incident}

PRIMARY ACCUSED:
{accused}

VICTIM:
{victim}

All findings require investigator verification before any action is taken.
"""

out_dir = r"d:\Projects\Criminal_detection\M1\data\firs"
for case in cases:
    ref, title, fir_no, date, station, accused, victim, incident = case
    content = template.format(
        fir_no=fir_no, date=date, station=station, ref=ref,
        incident=incident, accused=accused, victim=victim
    )
    with open(os.path.join(out_dir, f"{ref}.txt"), "w", encoding="utf-8") as f:
        f.write(content)

print(f"Generated {len(cases)} FIRs in {out_dir}")
