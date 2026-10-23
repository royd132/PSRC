"""Names of the six historical physical descriptor categories."""
FIELDS = (
    ("1-hour mean absolute 15-minute ramp", "3-hour mean absolute 15-minute ramp",
     "6-hour mean absolute 15-minute ramp",
     "3-interval net change", "11-interval net change"),
    ("power-to-reference ratio mean", "ratio variability", "adjacent ratio change",
     "ratio transition fraction", "last-daylight to first-daylight-quarter ratio difference"),
    ("24-hour correlation", "6-hour correlation", "one-step lag correlation"),
    ("24-hour temperature mean", "recent temperature departure", "temperature-power correlation"),
    ("daily sine", "daily cosine", "solar elevation proxy"),
    ("6-hour residual variability", "12-hour residual variability", "24-hour residual variability"),
)
