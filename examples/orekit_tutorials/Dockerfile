# ---------- Stage 1: build the Orekit tutorials with Maven ----------
FROM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /build

# Copy the pom first so Docker caches the dependency download.
# Rebuilds after a code change then skip re-downloading every jar.
COPY orekit-tutorials/pom.xml .
RUN mvn -q -B dependency:go-offline || true

# Copy the source and compile it, then gather every runtime jar into target/dependency
COPY orekit-tutorials/src ./src
RUN mvn -q -B -DskipTests compile dependency:copy-dependencies -DincludeScope=runtime

# ---------- Stage 2: small runtime image (no Maven, no source) ----------
FROM eclipse-temurin:21-jre
WORKDIR /app

COPY --from=build /build/target/classes ./classes
COPY --from=build /build/target/dependency ./lib

# Orekit's physical data (Earth orientation, leap seconds, JPL ephemerides).
# The tutorials look for a folder named "orekit-data" in the user's home
# directory, so we put it in /app and point user.home there.
COPY orekit-data ./orekit-data

ENTRYPOINT ["java", "-Duser.home=/app", "-cp", "/app/classes:/app/lib/*"]

# Default scenario. Override at run time to run a different tutorial, e.g.
#   docker run --rm halo-sim org.orekit.tutorials.propagation.cr3bp.ManifoldTransfer
CMD ["org.orekit.tutorials.propagation.cr3bp.EarthMoonHaloOrbit"]
