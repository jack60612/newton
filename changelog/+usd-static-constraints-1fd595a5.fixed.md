Keep USD CONNECT constraints with world, standalone zero-mass, or standalone world-fixed endpoints as native MuJoCo equalities under default import, avoiding unsupported orphan loop joints even when static hardware receives positive inferred collider mass.
Resolve static USD Xformable/site CONNECT endpoints to world while preserving their composed reference, scale, and import transforms.
Preserve static USD WELD endpoint rotations and local offsets in native relative poses, and load world site shapes independently of visual-loading flags for site-actuator references.
Discover every structural world-root branch and preserve its import placement and rebasing when a static support no longer contributes a rigid body.
