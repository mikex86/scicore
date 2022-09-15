@file:Suppress("INACCESSIBLE_TYPE")
import org.gradle.internal.os.OperatingSystem;

rootProject.name = "scicore"
include("matplotlib")
if (OperatingSystem.current() != OperatingSystem.MAC_OS) {
    include("cuda-backend")
}
include("core")
include("genericcpu-backend")
include("tests")
