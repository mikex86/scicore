plugins {
    id("java")
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven(url = "https://packages.jetbrains.team/maven/p/skija/maven")
}

val skiaArtifact = "skija-" + System.getProperty("os.name")!!.let {
    when {
        it.startsWith("Linux") -> "linux"
        it.startsWith("Mac OS X") || it.startsWith("Darwin") -> "macos-" + System.getProperty("os.arch")!!.let { arch ->
            when {
                arch.startsWith("aarch64") -> "arm64"
                else -> "x64"
            }
        }
        it.startsWith("Windows") -> "windows"
        else -> throw Error("Unrecognized or unsupported platform. Please set \"skiaArtifact\" manually")
    }
}

dependencies {
    implementation("org.jetbrains:annotations:23.0.0")

    implementation("org.jetbrains.skija:$skiaArtifact:0.93.1")

    testImplementation("org.junit.jupiter:junit-jupiter-api:5.9.0")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.9.0")
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}