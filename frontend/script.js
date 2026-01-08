let map;

function predict() {
  const city = document.getElementById("city").value;
  const lat = parseFloat(document.getElementById("lat").value);
  const lon = parseFloat(document.getElementById("lon").value);
  const hour = parseInt(document.getElementById("hour").value);
  const day = parseInt(document.getElementById("day").value);
  const traffic = parseFloat(document.getElementById("traffic").value);

  const resultBox = document.getElementById("result");
  const mapDiv = document.getElementById("map");

  if (
    isNaN(lat) || isNaN(lon) ||
    isNaN(hour) || isNaN(day) ||
    isNaN(traffic)
  ) {
    resultBox.style.display = "block";
    resultBox.className = "result high";
    resultBox.innerText = "Please fill all fields correctly.";
    return;
  }

  resultBox.style.display = "block";
  resultBox.className = "result";
  resultBox.innerText = "Analyzing traffic data...";

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      city: city,
      latitude: lat,
      longitude: lon,
      hour: hour,
      day: day,
      traffic_volume: traffic
    })
  })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        throw new Error(data.error);
      }

      const risk = data.accident_risk;
      const level = data.risk_level;

      if (level === "High") {
        resultBox.className = "result high";
        resultBox.innerText = `⚠️ HIGH RISK\nProbability: ${risk}`;
      } 
      else if(level=="Medium"){
        resultBox.className = "result medium";
        resultBox.innerText = `⚠️ MEDIUM RISK\nProbability: ${risk}`;
      }
      else {
        resultBox.className = "result low";
        resultBox.innerText = `✅ LOW RISK\nProbability: ${risk}`;
      }

      // Show map
      mapDiv.style.display = "block";

      if (map) {
        map.remove();
      }

      map = L.map("map").setView([lat, lon], 12);

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap"
      }).addTo(map);

      let color;
      if (level === "High") color = "red";
      else if (level === "Medium") color = "orange";
      else color = "green";

      L.circleMarker([lat, lon], {
        radius: 10,
        color: color,
        fillColor: color,
        fillOpacity: 0.8
      })
        .addTo(map)
        .bindPopup(
          `<b>${city || "Location"}</b><br>
           Risk: ${level}<br>
           Probability: ${risk}`
        )
        .openPopup();
    })
    .catch(err => {
      resultBox.className = "result high";
      resultBox.innerText = "Error connecting to backend.";
      console.error(err);
    });
}
