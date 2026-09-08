(function () {
  "use strict";

  var TERMINAL = { success: 1, failed: 1 };

  function fmtBytes(n) {
    if (n === null || n === undefined) return "—";
    var units = ["B", "KB", "MB", "GB", "TB"], i = 0, x = Number(n);
    while (x >= 1024 && i < units.length - 1) { x /= 1024; i++; }
    return (i === 0 ? x.toFixed(0) : x.toFixed(2)) + " " + units[i];
  }

  function fmtDate(ts) {
    return ts ? new Date(ts * 1000).toLocaleString() : "";
  }

  async function getJSON(url) {
    var r = await fetch(url, { headers: { Accept: "application/json" } });
    if (r.status === 401) { window.location.href = "/login"; return null; }
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }

  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  // ----- history table -----------------------------------------------------
  function renderList(jobs) {
    var body = document.getElementById("jobs-body");
    if (!body) return;
    body.textContent = "";
    jobs.forEach(function (j) {
      var tr = document.createElement("tr");
      var pct = Math.round((j.progress || 0) * 100);
      [
        fmtDate(j.created_at),
        j.model_name,
        j.instance_name
      ].forEach(function (val) {
        var td = document.createElement("td");
        td.textContent = val;
        tr.appendChild(td);
      });

      var tdStatus = document.createElement("td");
      var badge = document.createElement("span");
      badge.className = "badge badge-" + j.status;
      badge.textContent = j.status;
      tdStatus.appendChild(badge);
      tr.appendChild(tdStatus);

      var tdProg = document.createElement("td");
      tdProg.textContent = pct + "%";
      tr.appendChild(tdProg);

      var tdLink = document.createElement("td");
      var a = document.createElement("a");
      a.className = "link";
      a.href = "/jobs/" + j.id;
      a.textContent = "view";
      tdLink.appendChild(a);
      tr.appendChild(tdLink);

      body.appendChild(tr);
    });
  }

  async function pollList() {
    try {
      var data = await getJSON("/api/jobs");
      if (data) renderList(data.jobs || []);
    } catch (e) { /* keep trying */ }
    window.setTimeout(pollList, 4000);
  }

  // ----- job detail ------------------------------------------------------
  function infoRows(mi) {
    var rows = [];
    var d = mi.details || {};
    ["family", "format", "parameter_size", "quantization_level"].forEach(function (k) {
      if (d[k]) rows.push([k, String(d[k])]);
    });
    var m = mi.model_info || {};
    Object.keys(m).forEach(function (k) {
      if (/context_length|parameter_count|quantization|block_count|embedding_length/.test(k)) {
        rows.push([k, String(m[k])]);
      }
    });
    if (mi.digest) rows.push(["digest", String(mi.digest)]);
    return rows;
  }

  function renderJob(j) {
    var badge = document.getElementById("job-status");
    if (badge) { badge.textContent = j.status; badge.className = "badge badge-" + j.status; }
    setText("job-phase", j.phase || "");

    var bar = document.getElementById("job-bar");
    if (bar) bar.style.width = Math.round((j.progress || 0) * 100) + "%";

    setText("job-sha", j.gguf_sha256 || "—");
    setText("job-dlsize", j.download_size ? fmtBytes(j.download_size) : "—");
    setText("job-finalsize", j.final_size ? fmtBytes(j.final_size) : "—");
    setText("job-modelfile", j.modelfile || "—");

    var log = document.getElementById("job-log");
    if (log && Array.isArray(j.logs)) {
      log.textContent = j.logs.map(function (l) { return l.msg; }).join("\n");
    }

    var ok = document.getElementById("job-success");
    if (ok) ok.hidden = j.status !== "success";
    var err = document.getElementById("job-error");
    if (err) { err.hidden = j.status !== "failed"; err.textContent = j.error || ""; }
    var retry = document.getElementById("job-retry");
    if (retry) retry.hidden = j.status !== "failed";

    var info = document.getElementById("job-info");
    if (info && j.model_info) {
      var rows = infoRows(j.model_info);
      if (rows.length) {
        info.textContent = "";
        rows.forEach(function (pair) {
          var tr = document.createElement("tr");
          var th = document.createElement("th"); th.textContent = pair[0];
          var td = document.createElement("td"); td.className = "mono break"; td.textContent = pair[1];
          tr.appendChild(th); tr.appendChild(td);
          info.appendChild(tr);
        });
      }
    }
  }

  async function pollJob(id) {
    var delay = 2000;
    try {
      var j = await getJSON("/api/jobs/" + id);
      if (!j) return;
      renderJob(j);
      if (TERMINAL[j.status]) return;
    } catch (e) { delay = 5000; }
    window.setTimeout(function () { pollJob(id); }, delay);
  }

  // ----- ollama instance status (online check + version, shown in the <select>) --
  function renderInstances(list) {
    var select = document.getElementById("instance-select");
    if (!select) return;
    (list || []).forEach(function (inst) {
      var opt = select.querySelector('option[value="' + CSS.escape(inst.name) + '"]');
      if (!opt) return;
      var base = opt.dataset.baseLabel || opt.textContent;
      var status = inst.reachable
        ? "● online, v" + (inst.version || "?")
        : "○ unreachable";
      opt.textContent = base + "  —  " + status;
    });
  }

  async function pollInstances() {
    try {
      var data = await getJSON("/api/instances");
      if (data) renderInstances(data.instances || []);
    } catch (e) { /* keep last known labels, retry below */ }
    window.setTimeout(pollInstances, 15000);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var jobId = document.body.dataset.jobId;
    if (jobId) pollJob(jobId);
    if (document.querySelector("[data-jobs-list]")) pollList();
    if (document.getElementById("instance-select")) pollInstances();
  });
})();
