plugins {
    java
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

val lwjglVersion = "3.3.1"

val lwjglNatives = Pair(
    System.getProperty("os.name")!!,
    System.getProperty("os.arch")!!
).let { (name, arch) ->
    when {
        arrayOf("Linux", "FreeBSD", "SunOS", "Unit").any { name.startsWith(it) } ->
            if (arrayOf("arm", "aarch64").any { arch.startsWith(it) })
                "natives-linux${if (arch.contains("64") || arch.startsWith("armv8")) "-arm64" else "-arm32"}"
            else
                "natives-linux"
        arrayOf("Mac OS X", "Darwin").any { name.startsWith(it) } ->
            "natives-macos${if (arch.startsWith("aarch64")) "-arm64" else ""}"
        arrayOf("Windows").any { name.startsWith(it) } ->
            if (arch.contains("64"))
                "natives-windows${if (arch.startsWith("aarch64")) "-arm64" else ""}"
            else
                "natives-windows-x86"
        else -> throw Error("Unrecognized or unsupported platform. Please set \"lwjglNatives\" manually")
    }
}

dependencies {
    implementation("org.jetbrains:annotations:23.0.0")

    // LWJGL
    implementation(platform("org.lwjgl:lwjgl-bom:$lwjglVersion"))
    implementation("org.lwjgl", "lwjgl")
    implementation("org.lwjgl", "lwjgl-cuda")
    implementation("org.lwjgl", "lwjgl-jemalloc")
    runtimeOnly("org.lwjgl", "lwjgl", classifier = lwjglNatives)
    runtimeOnly("org.lwjgl", "lwjgl-jemalloc", classifier = lwjglNatives)

    testImplementation("org.junit.jupiter:junit-jupiter-api:5.6.0")
    testImplementation("org.junit.jupiter:junit-jupiter-params:5.8.1")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")
    testImplementation("me.tongfei:progressbar:0.9.3")
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}