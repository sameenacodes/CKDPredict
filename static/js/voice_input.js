(function () {
    "use strict";

    function initVoiceInput() {

        const SpeechRecognition =
            window.SpeechRecognition ||
            window.webkitSpeechRecognition;

        const form = document.querySelector("form");

        if (!form || !SpeechRecognition) {
            return;
        }

        // Prevent duplicate initialization
        if (form.dataset.voiceInitialized === "true") {
            return;
        }

        form.dataset.voiceInitialized = "true";

        const fields = [
            {
                name: "patient_name",
                aliases: ["patient name", "patient", "name"]
            },
            {
                name: "age",
                aliases: ["age"]
            },
            {
                name: "gender",
                aliases: ["gender", "sex"]
            },
            {
                name: "sc",
                aliases: ["serum creatinine", "creatinine"]
            },
            {
                name: "sg",
                aliases: ["specific gravity", "gravity"]
            },
            {
                name: "al",
                aliases: ["albumin"]
            },
            {
                name: "hemo",
                aliases: ["hemoglobin", "haemoglobin"]
            },
            {
                name: "rc",
                aliases: [
                    "red blood cell count",
                    "rbc count",
                    "rbc"
                ]
            },
            {
                name: "htn",
                aliases: [
                    "hypertension",
                    "blood pressure history"
                ]
            },
            {
                name: "dm",
                aliases: [
                    "diabetes mellitus",
                    "diabetes"
                ]
            },
            {
                name: "appet",
                aliases: ["appetite"]
            },
            {
                name: "pc",
                aliases: ["pus cell", "pus cells"]
            }
        ];

        function getField(name) {
            return form.querySelector(
                `[name="${name}"]`
            );
        }

        function setField(name, value) {

            const element = getField(name);

            if (!element || !value) {
                return false;
            }

            value = String(value).trim();

            if (element.tagName === "SELECT") {

                const option =
                    Array.from(element.options).find(
                        option =>
                            option.value.toLowerCase() ===
                                value.toLowerCase() ||
                            option.textContent
                                .trim()
                                .toLowerCase() ===
                                value.toLowerCase()
                    );

                if (!option) {
                    return false;
                }

                element.value = option.value;

            } else {

                element.value = value;
            }

            element.dispatchEvent(
                new Event("input", {
                    bubbles: true
                })
            );

            element.dispatchEvent(
                new Event("change", {
                    bubbles: true
                })
            );

            return true;
        }

        function escapeRegex(text) {

            return text.replace(
                /[.*+?^${}()|[\]\\]/g,
                "\\$&"
            );
        }

        const allAliases =
            fields.flatMap(field => field.aliases);

        function extractValue(text, field) {

            const aliases =
                [...field.aliases]
                    .sort(
                        (a, b) =>
                            b.length - a.length
                    )
                    .map(escapeRegex)
                    .join("|");

            const otherAliases =
                allAliases
                    .filter(
                        alias =>
                            !field.aliases.includes(alias)
                    )
                    .sort(
                        (a, b) =>
                            b.length - a.length
                    )
                    .map(escapeRegex)
                    .join("|");

            const pattern =
                `(?:${aliases})` +
                `\\s*(?:is|:|=|to)?\\s*` +
                `(.+?)` +
                `(?=\\s*(?:,|;|\\.|$|\\s+(?:${otherAliases})\\b))`;

            const regex =
                new RegExp(pattern, "i");

            const match =
                text.match(regex);

            return match
                ? match[1].trim()
                : null;
        }

        function parseVoiceText(transcript) {

            let text =
                transcript
                    .replace(
                        /\bpoint\b/gi,
                        "."
                    )
                    .replace(
                        /\bdecimal\b/gi,
                        "."
                    )
                    .replace(
                        /\band\b/gi,
                        ","
                    )
                    .replace(
                        /\s+/g,
                        " "
                    )
                    .trim();

            let count = 0;

            fields.forEach(field => {

                const value =
                    extractValue(
                        text,
                        field
                    );

                if (!value) {
                    return;
                }

                // Numeric fields
                if (
                    [
                        "age",
                        "sc",
                        "sg",
                        "al",
                        "hemo",
                        "rc"
                    ].includes(field.name)
                ) {

                    const number =
                        value
                            .replace(/,/g, ".")
                            .match(
                                /-?\d+(?:\.\d+)?/
                            );

                    if (
                        number &&
                        setField(
                            field.name,
                            number[0]
                        )
                    ) {
                        count++;
                    }

                    return;
                }

                // Gender
                if (field.name === "gender") {

                    if (
                        /\b(female|woman|girl)\b/i
                            .test(value)
                    ) {

                        if (
                            setField(
                                "gender",
                                "female"
                            )
                        ) {
                            count++;
                        }

                    } else if (
                        /\b(male|man|boy)\b/i
                            .test(value)
                    ) {

                        if (
                            setField(
                                "gender",
                                "male"
                            )
                        ) {
                            count++;
                        }
                    }

                    return;
                }

                // Hypertension / Diabetes
                if (
                    field.name === "htn" ||
                    field.name === "dm"
                ) {

                    if (
                        /\b(yes|present|have)\b/i
                            .test(value)
                    ) {

                        if (
                            setField(
                                field.name,
                                "yes"
                            )
                        ) {
                            count++;
                        }

                    } else if (
                        /\b(no|none|absent)\b/i
                            .test(value)
                    ) {

                        if (
                            setField(
                                field.name,
                                "no"
                            )
                        ) {
                            count++;
                        }
                    }

                    return;
                }

                // Appetite
                if (field.name === "appet") {

                    if (
                        /\b(poor|low)\b/i
                            .test(value)
                    ) {

                        if (
                            setField(
                                "appet",
                                "poor"
                            )
                        ) {
                            count++;
                        }

                    } else if (
                        /\b(good|normal)\b/i
                            .test(value)
                    ) {

                        if (
                            setField(
                                "appet",
                                "good"
                            )
                        ) {
                            count++;
                        }
                    }

                    return;
                }

                // Pus cell
                if (field.name === "pc") {

                    if (
                        /\babnormal\b/i
                            .test(value)
                    ) {

                        if (
                            setField(
                                "pc",
                                "abnormal"
                            )
                        ) {
                            count++;
                        }

                    } else if (
                        /\bnormal\b/i
                            .test(value)
                    ) {

                        if (
                            setField(
                                "pc",
                                "normal"
                            )
                        ) {
                            count++;
                        }
                    }

                    return;
                }

                // Patient name
                if (
                    field.name ===
                    "patient_name"
                ) {

                    if (
                        setField(
                            "patient_name",
                            value
                        )
                    ) {
                        count++;
                    }
                }

            });

            return count;
        }

        // Voice UI
        const box =
            document.createElement("div");

        box.id =
            "ckdVoiceInputBox";

        box.style.cssText =
            "margin:0 0 20px;" +
            "padding:16px;" +
            "border:1px solid #dbeafe;" +
            "background:#eff6ff;" +
            "border-radius:14px;" +
            "display:flex;" +
            "align-items:center;" +
            "gap:12px;" +
            "flex-wrap:wrap;";

        const button =
            document.createElement("button");

        button.type = "button";

        button.textContent =
            "🎙️ Voice Fill Assessment";

        button.style.cssText =
            "border:0;" +
            "background:#2563eb;" +
            "color:white;" +
            "padding:11px 16px;" +
            "border-radius:9px;" +
            "font-weight:700;" +
            "cursor:pointer;";

        const language =
            document.createElement("select");

        language.innerHTML =
            `
            <option value="en-IN">
                English
            </option>
            <option value="ta-IN">
                தமிழ்
            </option>
            `;

        language.style.cssText =
            "padding:10px;" +
            "border:1px solid #d1d5db;" +
            "border-radius:9px;";

        const status =
            document.createElement("span");

        status.textContent =
            "Speak patient details and clinical values.";

        status.style.cssText =
            "color:#4b5563;font-size:12px;";

        box.append(
            button,
            language,
            status
        );

        form.parentNode.insertBefore(
            box,
            form
        );

        const recognition =
            new SpeechRecognition();

        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        let listening = false;

        button.addEventListener(
            "click",
            function (event) {

                event.preventDefault();
                event.stopPropagation();

                if (listening) {
                    return;
                }

                listening = true;

                recognition.lang =
                    language.value;

                status.textContent =
                    "🎙️ Listening... Speak now.";

                button.textContent =
                    "⏹️ Listening...";

                button.disabled = true;

                try {

                    recognition.start();

                } catch (error) {

                    listening = false;

                    button.disabled = false;

                    button.textContent =
                        "🎙️ Voice Fill Assessment";

                    status.textContent =
                        "Microphone could not start. Click again.";
                }
            }
        );

        recognition.onresult =
            function (event) {

                const transcript =
                    event.results[0][0]
                        .transcript;

                const filled =
                    parseVoiceText(
                        transcript
                    );

                status.textContent =
                    `Heard: "${transcript}" — ${filled} field(s) filled.`;
            };

        recognition.onerror =
            function (event) {

                const messages = {

                    "not-allowed":
                        "Microphone permission denied.",

                    "service-not-allowed":
                        "Speech recognition service is blocked.",

                    "no-speech":
                        "No speech detected. Try again.",

                    "audio-capture":
                        "No microphone detected.",

                    "network":
                        "Speech recognition network error."
                };

                status.textContent =
                    messages[event.error] ||
                    `Voice input error: ${event.error}`;
            };

        recognition.onend =
            function () {

                listening = false;

                button.disabled = false;

                button.textContent =
                    "🎙️ Voice Fill Assessment";
            };
    }

    if (
        document.readyState ===
        "loading"
    ) {

        document.addEventListener(
            "DOMContentLoaded",
            initVoiceInput,
            { once: true }
        );

    } else {

        initVoiceInput();
    }

})();