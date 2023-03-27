from pbim_preprocessor.assembler.util import MergeChannelsConfig
from pbim_preprocessor.parser.pbim import (
    POST_PROCESSABLE_CHANNELS,
    PBimArtificialDataParser,
)
from pbim_preprocessor.processor.pbim import PBimArtificialDataProcessor
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
    "pbim": PBimArtificialDataProcessor.CHANNELS,
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
        "239T",
        "240T",
        "511V",
        "521T",
        "315V",
        "208V",
        "421L",
        "239V",
        "431L",
        "321V",
        "102V",
        "113V",
        "532V",
        "129V",
        "512L",
        "218L",
        "542L",
        "306V",
        "233V",
        "531T",
        "199V",
        "208T",
        "236T",
        "240V",
        "223T",
        "200T",
        "334V",
        "238V",
        "143V",
        "123V",
        "311V",
        "225T",
        "441L",
        "213T",
        "431V",
        "122V",
        "305V",
        "228V",
        "303V",
        "411V",
        "222T",
        "212V",
        "231T",
        "204V",
        "312V",
        "234L",
        "541L",
        "221V",
        "203V",
        "201T",
        "209L",
        "531L",
        "211T",
        "133V",
        "229T",
        "230T",
        "201V",
        "411L",
        "238L",
        "199T",
        "422V",
        "542T",
        "228L",
        "442L",
        "216T",
        "299V",
        "103V",
        "310V",
        "119V",
        "320V",
        "522V",
        "202V",
        "432V",
        "442T",
        "216V",
        "512T",
        "204T",
        "223V",
        "412T",
        "301V",
        "228T",
        "335V",
        "124V",
        "441V",
        "105V",
        "421V",
        "235V",
        "442V",
        "227V",
        "329V",
        "521L",
        "432L",
        "336V",
        "238T",
        "224T",
        "511L",
        "205V",
        "327V",
        "112V",
        "317V",
        "202T",
        "323V",
        "322V",
        "209V",
        "200V",
        "219T",
        "541V",
        "204L",
        "118V",
        "333V",
        "139V",
        "521V",
        "224L",
        "242T",
        "342V",
        "422T",
        "328V",
        "199L",
        "338V",
        "307V",
        "412L",
        "422L",
        "217V",
        "229V",
        "239L",
        "132V",
        "225V",
        "232T",
        "243T",
        "341V",
        "232V",
        "126V",
        "101V",
        "531V",
        "100V",
        "235T",
        "325V",
        "120V",
        "222V",
        "104V",
        "339V",
        "332V",
        "207V",
        "308V",
        "214T",
        "411T",
        "242V",
        "220V",
        "532T",
        "212T",
        "412V",
        "324V",
        "206V",
        "326V",
        "330V",
        "99V",
        "221T",
        "234T",
        "207T",
        "241V",
        "304V",
        "316V",
        "127V",
        "203T",
        "210V",
        "343V",
        "511T",
        "206T",
        "214L",
        "522T",
        "522L",
        "243L",
        "233L",
        "110V",
        "229L",
        "213V",
        "128V",
        "241T",
        "220T",
        "210T",
        "431T",
        "313V",
        "106V",
        "231V",
        "213L",
        "318V",
        "121V",
        "331V",
        "135V",
        "337V",
        "136V",
        "208L",
        "211V",
        "219V",
        "109V",
        "314V",
        "236V",
        "115V",
        "532L",
        "114V",
        "309V",
        "107V",
        "116V",
        "227T",
        "226V",
        "319V",
        "421T",
        "217T",
        "137V",
        "542V",
        "131V",
        "218V",
        "117V",
        "215V",
        "125V",
        "140V",
        "512V",
        "219L",
        "234V",
        "209T",
        "111V",
        "134V",
        "300V",
        "130V",
        "218T",
        "223L",
        "226T",
        "432T",
        "215T",
        "142V",
        "340V",
        "541T",
        "214V",
        "237V",
        "302V",
        "138V",
        "441T",
        "205T",
        "141V",
        "237T",
        "224V",
        "108V",
        "243V",
        "203L",
    ],
    "z24-pdt-fvt": [
        "343V",
        "211V",
        "106V",
        "326V",
        "114V",
        "207V",
        "119V",
        "330V",
        "239V",
        "422T",
        "214L",
        "412V",
        "218T",
        "201T",
        "117V",
        "531L",
        "312V",
        "227T",
        "120V",
        "109V",
        "104V",
        "322V",
        "203T",
        "341V",
        "140V",
        "304V",
        "299V",
        "522V",
        "133V",
        "210V",
        "229L",
        "132V",
        "521L",
        "511L",
        "122V",
        "225T",
        "208L",
        "307V",
        "128V",
        "142V",
        "213T",
        "340V",
        "421L",
        "541L",
        "541V",
        "212T",
        "218V",
        "216T",
        "101V",
        "512T",
        "411L",
        "223T",
        "333V",
        "237V",
        "220V",
        "522T",
        "214T",
        "219T",
        "521V",
        "213V",
        "412L",
        "209V",
        "222V",
        "240V",
        "411T",
        "214V",
        "531T",
        "215V",
        "315V",
        "305V",
        "102V",
        "230V",
        "331V",
        "130V",
        "240T",
        "110V",
        "226V",
        "336V",
        "129V",
        "243L",
        "233V",
        "224L",
        "219L",
        "202T",
        "205V",
        "209L",
        "217V",
        "324V",
        "216V",
        "235V",
        "134V",
        "225V",
        "233T",
        "228L",
        "531V",
        "511T",
        "203V",
        "103V",
        "125V",
        "431L",
        "241T",
        "229T",
        "532T",
        "421V",
        "205T",
        "218L",
        "441L",
        "139V",
        "432T",
        "219V",
        "118V",
        "213L",
        "532V",
        "136V",
        "243V",
        "236T",
        "442V",
        "311V",
        "332V",
        "204T",
        "238V",
        "121V",
        "432L",
        "231T",
        "228V",
        "441V",
        "342V",
        "239L",
        "224T",
        "199T",
        "422L",
        "221V",
        "238T",
        "335V",
        "542V",
        "206V",
        "211T",
        "313V",
        "308V",
        "431V",
        "314V",
        "233L",
        "522L",
        "206T",
        "232V",
        "328V",
        "100V",
        "320V",
        "339V",
        "224V",
        "227V",
        "241V",
        "542T",
        "127V",
        "115V",
        "442T",
        "208T",
        "105V",
        "337V",
        "432V",
        "442L",
        "228T",
        "229V",
        "217T",
        "422V",
        "202V",
        "235T",
        "223V",
        "237T",
        "541T",
        "207T",
        "208V",
        "327V",
        "112V",
        "542L",
        "210T",
        "441T",
        "411V",
        "316V",
        "222T",
        "239T",
        "421T",
        "243T",
        "223L",
        "141V",
        "300V",
        "317V",
        "238L",
        "113V",
        "301V",
        "302V",
        "131V",
        "309V",
        "230T",
        "220T",
        "107V",
        "204V",
        "201V",
        "511V",
        "143V",
        "321V",
        "212V",
        "199V",
        "412T",
        "318V",
        "310V",
        "323V",
        "200T",
        "303V",
        "532L",
        "338V",
        "116V",
        "234T",
        "431T",
        "234L",
        "108V",
        "126V",
        "306V",
        "521T",
        "231V",
        "111V",
        "325V",
        "329V",
        "319V",
        "135V",
        "137V",
        "209T",
        "200V",
        "242T",
        "242V",
        "124V",
        "226T",
        "236V",
        "123V",
        "215T",
        "234V",
        "138V",
        "232T",
        "221T",
        "334V",
        "204L",
    ],
    "lux": [
        "30185",
        "21725",
        "30187",
        "29701",
        "30188",
        "29700",
        "30190",
        "21728",
        "30498",
        "21916",
        "30499",
        "21917",
        "30500",
        "21919",
        "30501",
        "30189",
        "39376",
        "39375",
        "39371",
        "39373",
        "39369",
        "39370",
        "39372",
        "39374",
        "30183",
        "29702",
        # "21921",  This is missing from some measurements in the dataset
        "Temperature",
    ],
}
MERGE_CONFIGS = {
    "z24-ems": [MergeChannelsConfig(["TBC1", "TBC2"], "TBC")],
    "z24-pdt": [],
    "pbim": [],
    "grandstand": [],
    "lux": [],
}
