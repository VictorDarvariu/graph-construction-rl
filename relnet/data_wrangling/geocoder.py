import json

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

class Geocoder(object):
    NOMINATIM_USER_AGENT = "Purpose: academic research; Contact email: v.darvariu@ucl.ac.uk"

    COUNTRY_CODES_EUROPE = ["al","ad","at","az","by","be","ba","bg","hr","cy","cz","dk","ee","fi","fr","ge","de","gr",
                            "hu","is","ie", "it","kz","xk","lv","li","lt","lu","mk","mt","md","mc","me","nl","no","pl",
                            "pt","ro","ru","sm","rs","sk", "si","es","se","ch","tr","ua","gb","va"]

    # see https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2#XK
    COUNTRY_CODE_TO_NAME = {"al": "Albania",
                            "ad": "Andorra",
                            "at" : "Austria",
                            "az": "Azeribaijan",
                            "by": "Belarus",
                            "be": "Belgium",
                            "ba": "Bosnia and Herzegovina",
                            "bg": "Bulgaria",
                            "hr": "Croatia",
                            "cy": "Cyprus",
                            "cz": "Czech Republic",
                            "dk": "Denmark",
                            "ee": "Estonia",
                            "fi": "Finland",
                            "fr": "France",
                            "ge": "Georgia",
                            "de": "Germany",
                            "gr" : "Greece",
                            "hu": "Hungary",
                            "is": "Iceland",
                            "ie": "Ireland",
                            "it": "Italy",
                            "kz": "Kazakhstan",
                            "xk": "Kosovo",
                            "lv": "Latvia",
                            "li": "Liechtenstein",
                            "lt": "Lithuania",
                            "lu": "Luxembourg",
                            "mk": "North Macedonia",
                            "mt": "Malta",
                            "md": "Moldova, Republic of",
                            "mc": "Monaco",
                            "me": "Montenegro",
                            "nl": "Netherlands",
                            "no": "Norway",
                            "pl": "Poland",
                            "pt": "Portugal",
                            "ro": "Romania",
                            "ru": "Russia",
                            "sm": "San Marino",
                            "rs": "Serbia",
                            "sk": "Slovakia",
                            "si": "Slovenia",
                            "es": "Spain",
                            "se": "Sweden",
                            "ch": "Switzerland",
                            "tr" : "Turkey",
                            "ua": "Ukraine",
                            "gb": "United Kingdom",
                            "va": "Holy See"}

    def __init__(self, raw_dataset_dir, dataset_name):
        self.raw_dataset_dir = raw_dataset_dir
        self.dataset_name = dataset_name

        self.geocoded_file_name = f"{self.dataset_name}_geocoded.json"
        self.geocoded_data_file = self.raw_dataset_dir / self.geocoded_file_name

        self.geolocator = Nominatim(user_agent=self.NOMINATIM_USER_AGENT)

        rate_limiter_kwargs = {'min_delay_seconds': 1, 'max_retries': 10, 'error_wait_seconds':10.0}

        self.geocode = RateLimiter(self.geolocator.geocode, **rate_limiter_kwargs)
        self.reverse = RateLimiter(self.geolocator.reverse, **rate_limiter_kwargs)

    def geocode_location(self, location_string, restrict_to_europe=True):
        location = self.geocode(location_string, language='en', addressdetails=True, country_codes=(self.COUNTRY_CODES_EUROPE if restrict_to_europe else None))
        if location is None:
            print(f"could not geocode location {location_string}!")
        return location

    def reverse_geocode_location(self, lat_lon_tuple):
        location = self.reverse(lat_lon_tuple)
        if location is None:
            print(f"could not geocode coords {lat_lon_tuple}!")
        return location

    def exists_geocoded_data(self):
        return self.geocoded_data_file.exists()

    def read_geocoded_data(self):
        with open(self.geocoded_data_file.resolve(), "r") as fh:
            json_data = json.load(fh)
            print(f"read in geolocation json data")
            return json_data

    def write_geocoded_data(self, geocoded_data):
        with open(self.geocoded_data_file.resolve(), "w") as fh:
            json.dump(geocoded_data, fh, indent=4)
        return geocoded_data