from pbim_preprocessor.assembler import MergeChannelsConfig
from pbim_preprocessor.parser import POST_PROCESSABLE_CHANNELS
from pbim_preprocessor.sampling import (
    MeanSamplingStrategy,
    LinearInterpolationSamplingStrategy,
)
from pbim_preprocessor.writer import CsvWriter, BinaryWriter

STRATEGIES = {
    "mean": MeanSamplingStrategy(),
    "interpolate": LinearInterpolationSamplingStrategy(),
}
FORMATS = {
    "csv": CsvWriter,
    "binary": BinaryWriter,
}
CHANNELS = {
    "pbim": POST_PROCESSABLE_CHANNELS,
    "grandstand": [f"Joint {i}" for i in range(1, 31)],
    "z24-ems": [
        "WS",
        "WD",
        "AT",
        "R",
        "H",
        "TE",
        "ADU",
        "ADK",
        "TSPU1",
        "TSPU2",
        "TSPU3",
        "TSAU1",
        "TSAU2",
        "TSAU3",
        "TSPK1",
        "TSPK2",
        "TSPK3",
        "TSAK1",
        "TSAK2",
        "TSAK3",
        "TBC1",
        "TBC2",
        "TSWS1",
        "TSWN1",
        "TWS1",
        "TWC1",
        "TWN1",
        "TP1",
        "TDT1",
        "TDS1",
        "TS1",
        "TSWS2",
        "TSWN2",
        "TWS2",
        "TWC2",
        "TWN2",
        "TP2",
        "TDT2",
        "TDS2",
        "TS2",
        "TWS3",
        "TWN3",
        "TWC3",
        "TP3",
        "TDT3",
        "TS3",
        "03",
        "05",
        "06",
        "07",
        "10",
        "12",
        "14",
        "16",
    ],
    "z24-pdt-avt": [
        "99V",
        "100V",
        "101V",
        "102V",
        "103V",
        "199L",
        "199T",
        "199V",
        "200T",
        "200V",
        "201T",
        "201V",
        "202T",
        "202V",
        "203L",
        "203T",
        "203V",
        "299V",
        "300V",
        "301V",
        "302V",
        "303V",
        "511L",
        "511T",
        "511V",
        "512L",
        "512T",
        "512V",
        "R1V",
        "R2L",
        "R2T",
        "R2V",
        "R3V",
        "104V",
        "105V",
        "106V",
        "107V",
        "108V",
        "204L",
        "204T",
        "204V",
        "205T",
        "205V",
        "206T",
        "206V",
        "207T",
        "207V",
        "208L",
        "208T",
        "208V",
        "304V",
        "305V",
        "306V",
        "307V",
        "308V",
        "521L",
        "521T",
        "521V",
        "522L",
        "522T",
        "522V",
        "R1V",
        "R2L",
        "R2T",
        "R2V",
        "R3V",
        "109V",
        "110V",
        "111V",
        "112V",
        "113V",
        "209L",
        "209T",
        "209V",
        "210T",
        "210V",
        "211T",
        "211V",
        "212T",
        "212V",
        "213L",
        "213T",
        "213V",
        "309V",
        "310V",
        "311V",
        "312V",
        "313V",
        "531L",
        "531T",
        "531V",
        "532L",
        "532T",
        "532V",
        "R1V",
        "R2L",
        "R2T",
        "R2V",
        "R3V",
        "114V",
        "115V",
        "116V",
        "117V",
        "118V",
        "214L",
        "214T",
        "214V",
        "215T",
        "215V",
        "216T",
        "216V",
        "217T",
        "217V",
        "218L",
        "218T",
        "218V",
        "314V",
        "315V",
        "316V",
        "317V",
        "318V",
        "541L",
        "541T",
        "541V",
        "542L",
        "542T",
        "542V",
        "R1V",
        "R2L",
        "R2T",
        "R2V",
        "R3V",
        "119V",
        "120V",
        "121V",
        "122V",
        "123V",
        "219L",
        "219T",
        "219V",
        "220T",
        "220V",
        "221T",
        "221V",
        "222T",
        "222V",
        "223L",
        "223T",
        "223V",
        "319V",
        "320V",
        "321V",
        "322V",
        "323V",
        "R1V",
        "R2L",
        "R2T",
        "R2V",
        "R3V",
        "124V",
        "125V",
        "126V",
        "127V",
        "128V",
        "224L",
        "224T",
        "224V",
        "225T",
        "225V",
        "226T",
        "226V",
        "227T",
        "227V",
        "228L",
        "228T",
        "228V",
        "324V",
        "325V",
        "326V",
        "327V",
        "328V",
        "411L",
        "411T",
        "411V",
        "412L",
        "412T",
        "412V",
        "R1V",
        "R2L",
        "R2T",
        "R2V",
        "R3V",
        "129V",
        "130V",
        "131V",
        "132V",
        "133V",
        "229L",
        "229T",
        "229V",
        "230T",
        "231T",
        "231V",
        "232T",
        "232V",
        "233L",
        "233V",
        "329V",
        "330V",
        "331V",
        "332V",
        "333V",
        "421L",
        "421T",
        "421V",
        "422L",
        "422T",
        "422V",
        "R1V",
        "R2L",
        "R2T",
        "R2V",
        "R3V",
        "134V",
        "135V",
        "136V",
        "137V",
        "138V",
        "234L",
        "234T",
        "234V",
        "235T",
        "235V",
        "236T",
        "236V",
        "237T",
        "237V",
        "238L",
        "238T",
        "238V",
        "334V",
        "335V",
        "336V",
        "337V",
        "338V",
        "431L",
        "431T",
        "431V",
        "432L",
        "432T",
        "432V",
        "R1V",
        "R2L",
        "R2T",
        "R2V",
        "R3V",
        "139V",
        "140V",
        "141V",
        "142V",
        "143V",
        "239L",
        "239T",
        "239V",
        "240T",
        "240V",
        "241T",
        "241V",
        "242T",
        "242V",
        "243L",
        "243T",
        "243V",
        "339V",
        "340V",
        "341V",
        "342V",
        "343V",
        "441L",
        "441T",
        "441V",
        "442L",
        "442T",
        "442V",
        "R1V",
        "R2L",
        "R2T",
        "R2V",
        "R3V",
    ],
    "z24-pdt-fvt": [
        "100V",
        "101V",
        "102V",
        "103V",
        "104V",
        "105V",
        "106V",
        "107V",
        "108V",
        "109V",
        "110V",
        "111V",
        "112V",
        "113V",
        "114V",
        "115V",
        "116V",
        "117V",
        "118V",
        "119V",
        "120V",
        "121V",
        "122V",
        "123V",
        "124V",
        "125V",
        "126V",
        "127V",
        "128V",
        "129V",
        "130V",
        "131V",
        "132V",
        "133V",
        "134V",
        "135V",
        "136V",
        "137V",
        "138V",
        "139V",
        "140V",
        "141V",
        "142V",
        "143V",
        "199T",
        "199V",
        "200T",
        "200V",
        "201T",
        "201V",
        "202T",
        "202V",
        "203T",
        "203V",
        "204L",
        "204T",
        "204V",
        "205T",
        "205V",
        "206T",
        "206V",
        "207T",
        "207V",
        "208L",
        "208T",
        "208V",
        "209L",
        "209T",
        "209V",
        "210T",
        "210V",
        "211T",
        "211V",
        "212T",
        "212V",
        "213L",
        "213T",
        "213V",
        "214L",
        "214T",
        "214V",
        "215T",
        "215V",
        "216T",
        "216V",
        "217T",
        "217V",
        "218L",
        "218T",
        "218V",
        "219L",
        "219T",
        "219V",
        "220T",
        "220V",
        "221T",
        "221V",
        "222T",
        "222V",
        "223L",
        "223T",
        "223V",
        "224L",
        "224T",
        "224V",
        "225T",
        "225V",
        "226T",
        "226V",
        "227T",
        "227V",
        "228L",
        "228T",
        "228V",
        "229L",
        "229T",
        "229V",
        "230T",
        "230V",
        "231T",
        "231V",
        "232T",
        "232V",
        "233L",
        "233T",
        "233V",
        "234L",
        "234T",
        "234V",
        "235T",
        "235V",
        "236T",
        "236V",
        "237T",
        "237V",
        "238L",
        "238T",
        "238V",
        "239L",
        "239T",
        "239V",
        "240T",
        "240V",
        "241T",
        "241V",
        "242T",
        "242V",
        "243L",
        "243T",
        "243V",
        "299V",
        "300V",
        "301V",
        "302V",
        "303V",
        "304V",
        "305V",
        "306V",
        "307V",
        "308V",
        "309V",
        "310V",
        "311V",
        "312V",
        "313V",
        "314V",
        "315V",
        "316V",
        "317V",
        "318V",
        "319V",
        "320V",
        "321V",
        "322V",
        "323V",
        "324V",
        "325V",
        "326V",
        "327V",
        "328V",
        "329V",
        "330V",
        "331V",
        "332V",
        "333V",
        "334V",
        "335V",
        "336V",
        "337V",
        "338V",
        "339V",
        "340V",
        "341V",
        "342V",
        "343V",
        "411L",
        "411T",
        "411V",
        "412L",
        "412T",
        "412V",
        "421L",
        "421T",
        "421V",
        "422L",
        "422T",
        "422V",
        "431L",
        "431T",
        "431V",
        "432L",
        "432T",
        "432V",
        "441L",
        "441T",
        "441V",
        "442L",
        "442T",
        "442V",
        "511L",
        "511T",
        "511V",
        "512T",
        "521L",
        "521T",
        "521V",
        "522L",
        "522T",
        "522V",
        "531L",
        "531T",
        "531V",
        "532L",
        "532T",
        "532V",
        "541L",
        "541T",
        "541V",
        "542L",
        "542T",
        "542V",
        "99V",
        "DP1V",
        "DP2V",
        "R1V",
        "R2L",
        "R2T",
        "R2V",
        "R3V",
    ],
}
MERGE_CONFIGS = {
    "z24-ems": [MergeChannelsConfig(["TBC1", "TBC2"], "TBC")],
    "z24-pdt": [],
    "pbim": [],
    "grandstand": [],
}
