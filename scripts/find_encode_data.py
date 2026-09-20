import requests

url = "https://www.encodeproject.org/search/"

params = {
    "type": "Experiment",
    "assay_title": "Histone ChIP-seq",
    "status": "released",
    "organism.scientific_name": "Homo sapiens",
    "format": "json"
}

response = requests.get(
    url,
    params=params,
    headers={"accept": "application/json"}
)

print(response.url)
print(response.status_code)

if response.status_code == 200:
    data = response.json()
    print(len(data["@graph"]))