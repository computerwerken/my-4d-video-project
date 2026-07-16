// jg4d vitrine prep - Photoshop companion
// ---------------------------------------
// File > Scripts > Browse... > this file.
//
// Round-trips vitrine_prep.py's output through Photoshop for hand work:
//
//   BUILD    pick a *_albedo.png -> a layered .psd with the matte as a real
//            layer mask, the ink map and the glass scuff as layers above it.
//   EXPORT   with that .psd open -> writes the four PNGs back out under the
//            same names, so Blender picks the edits up with no re-linking.
//
// The python does the parts Photoshop is bad at (FFT descreen, flat-field fit,
// screen analysis). This does the parts Photoshop is good at, which is you,
// with a brush, deciding where the wipe marks go.
//
// Tested against the file naming vitrine_prep.py emits: <stem>_albedo.png,
// <stem>_alpha.png, <stem>_ink.png, <stem>_scuff.png, <stem>.json
//
// MIT License.

#target photoshop

var LAYER_INK = "ink (density -> dots/bump/gloss)";
var LAYER_SCUFF = "glass scuff (white = worn)";
var LAYER_ITEM = "item";

function sid(s) { return stringIDToTypeID(s); }

function targetChannel(which) {
    var r = new ActionReference();
    r.putEnumerated(sid("channel"), sid("channel"), sid(which));
    var d = new ActionDescriptor();
    d.putReference(sid("null"), r);
    d.putBoolean(sid("makeVisible"), false);
    executeAction(sid("select"), d, DialogModes.NO);
}

function addRevealAllMask() {
    var d = new ActionDescriptor();
    d.putClass(sid("new"), sid("channel"));
    var r = new ActionReference();
    r.putEnumerated(sid("channel"), sid("channel"), sid("mask"));
    d.putReference(sid("at"), r);
    d.putEnumerated(sid("using"), sid("userMaskEnabled"), sid("revealAll"));
    executeAction(sid("make"), d, DialogModes.NO);
}

function stemOf(name) {
    var m = name.match(/^(.*)_albedo\.[^.]+$/i);
    if (m) return m[1];
    return name.replace(/\.[^.]+$/, "");
}

function sibling(folder, stem, kind) {
    var exts = ["png", "PNG", "tif", "tiff"];
    for (var i = 0; i < exts.length; i++) {
        var f = new File(folder + "/" + stem + "_" + kind + "." + exts[i]);
        if (f.exists) return f;
    }
    return null;
}

// ---------------------------------------------------------------------------
// BUILD
// ---------------------------------------------------------------------------

function build() {
    var alb = File.openDialog("Pick a *_albedo.png from vitrine_prep.py", "*.png;*.tif");
    if (!alb) return;

    var folder = alb.parent.fsName;
    var stem = stemOf(decodeURI(alb.name));

    var doc = app.open(alb);
    doc.activeLayer.isBackgroundLayer = false;
    doc.activeLayer.name = LAYER_ITEM;

    var prevUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    // --- matte -> a real layer mask on the item -----------------------------
    var alphaFile = sibling(folder, stem, "alpha");
    if (alphaFile) {
        var ad = app.open(alphaFile);
        ad.selection.selectAll();
        ad.selection.copy();
        ad.close(SaveOptions.DONOTSAVECHANGES);

        app.activeDocument = doc;
        doc.activeLayer = doc.artLayers.getByName(LAYER_ITEM);
        addRevealAllMask();
        targetChannel("mask");
        doc.paste();
        targetChannel("RGB");
        doc.selection.deselect();
    }

    // --- ink + scuff as plain layers, off by default -------------------------
    var extras = [
        { kind: "ink", name: LAYER_INK, mode: NORMAL },
        { kind: "scuff", name: LAYER_SCUFF, mode: NORMAL }
    ];
    for (var i = 0; i < extras.length; i++) {
        var f = sibling(folder, stem, extras[i].kind);
        if (!f) continue;
        var ed = app.open(f);
        ed.activeLayer.isBackgroundLayer = false;
        var dup = ed.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
        ed.close(SaveOptions.DONOTSAVECHANGES);
        app.activeDocument = doc;
        dup.name = extras[i].name;
        dup.visible = false;
    }

    app.preferences.rulerUnits = prevUnits;

    var out = new File(folder + "/" + stem + "_vitrine.psd");
    var opt = new PhotoshopSaveOptions();
    opt.layers = true;
    opt.embedColorProfile = true;
    doc.saveAs(out, opt, false, Extension.LOWERCASE);

    alert("Built " + out.name + "\n\n" +
          "item      - the descreened scan, matte already applied as a layer mask\n" +
          "ink       - density. Paint here to change where dots and relief appear.\n" +
          "glass scuff - white = worn. Paint long arcs, a few hard scratches,\n" +
          "              one or two faint smudges. Low opacity, soft round brush.\n\n" +
          "When you are done: run this script again with the .psd open and pick\n" +
          "Export. It writes the four PNGs back under the same names, so Blender\n" +
          "picks the edits up without re-linking anything.");
}

// ---------------------------------------------------------------------------
// EXPORT
// ---------------------------------------------------------------------------

function exportOne(doc, folder, stem, kind, layerName, grayscale) {
    var tmp;
    try {
        tmp = doc.duplicate(stem + "_tmp", false);
    } catch (e) { return false; }

    app.activeDocument = tmp;
    var found = false;
    for (var i = tmp.artLayers.length - 1; i >= 0; i--) {
        var L = tmp.artLayers[i];
        if (L.name === layerName) { L.visible = true; found = true; }
        else { try { L.remove(); } catch (e2) {} }
    }
    if (!found) { tmp.close(SaveOptions.DONOTSAVECHANGES); return false; }

    if (grayscale) tmp.changeMode(ChangeMode.GRAYSCALE);
    tmp.flatten();
    var f = new File(folder + "/" + stem + "_" + kind + ".png");
    var o = new PNGSaveOptions();
    o.compression = 6;
    o.interlaced = false;
    tmp.saveAs(f, o, true, Extension.LOWERCASE);
    tmp.close(SaveOptions.DONOTSAVECHANGES);
    return true;
}

function exportAll() {
    if (!app.documents.length) { alert("Open the _vitrine.psd first."); return; }
    var doc = app.activeDocument;
    var name = decodeURI(doc.name);
    if (!/_vitrine\.psd$/i.test(name)) {
        if (!confirm("This does not look like a _vitrine.psd. Export anyway?")) return;
    }
    var folder;
    try { folder = doc.path.fsName; }
    catch (e) { alert("Save the .psd somewhere first."); return; }
    var stem = name.replace(/_vitrine\.psd$/i, "").replace(/\.psd$/i, "");

    var wrote = [];

    // albedo: the item layer, mask applied
    var t = doc.duplicate(stem + "_alb", false);
    app.activeDocument = t;
    for (var i = t.artLayers.length - 1; i >= 0; i--) {
        var L = t.artLayers[i];
        if (L.name === LAYER_ITEM) L.visible = true;
        else { try { L.remove(); } catch (e) {} }
    }
    t.flatten();
    var o = new PNGSaveOptions(); o.compression = 6;
    t.saveAs(new File(folder + "/" + stem + "_albedo.png"), o, true, Extension.LOWERCASE);
    t.close(SaveOptions.DONOTSAVECHANGES);
    wrote.push("albedo");

    // alpha: the item's layer mask, as its own grayscale file
    app.activeDocument = doc;
    try {
        var t2 = doc.duplicate(stem + "_alp", false);
        app.activeDocument = t2;
        var item = t2.artLayers.getByName(LAYER_ITEM);
        t2.activeLayer = item;
        // rasterise the mask into pixels: fill the layer white, apply the mask
        item.visible = true;
        for (var j = t2.artLayers.length - 1; j >= 0; j--) {
            if (t2.artLayers[j].name !== LAYER_ITEM) {
                try { t2.artLayers[j].remove(); } catch (e3) {}
            }
        }
        t2.selection.selectAll();
        var white = new SolidColor(); white.rgb.red = 255; white.rgb.green = 255; white.rgb.blue = 255;
        t2.selection.fill(white);
        t2.selection.deselect();
        var bg = t2.artLayers.add();
        bg.name = "black";
        bg.move(t2.artLayers.getByName(LAYER_ITEM), ElementPlacement.PLACEAFTER);
        t2.activeLayer = bg;
        t2.selection.selectAll();
        var black = new SolidColor(); black.rgb.red = 0; black.rgb.green = 0; black.rgb.blue = 0;
        t2.selection.fill(black);
        t2.selection.deselect();
        t2.flatten();
        t2.changeMode(ChangeMode.GRAYSCALE);
        t2.saveAs(new File(folder + "/" + stem + "_alpha.png"), o, true, Extension.LOWERCASE);
        t2.close(SaveOptions.DONOTSAVECHANGES);
        wrote.push("alpha");
    } catch (e) { /* no mask on the item: leave the existing alpha alone */ }

    app.activeDocument = doc;
    if (exportOne(doc, folder, stem, "ink", LAYER_INK, true)) wrote.push("ink");
    app.activeDocument = doc;
    if (exportOne(doc, folder, stem, "scuff", LAYER_SCUFF, true)) wrote.push("scuff");

    app.activeDocument = doc;
    alert("Wrote: " + wrote.join(", ") + "\n\nIn Blender, hit Build vitrine again\n" +
          "(or just reload the images) to see the changes.");
}

// ---------------------------------------------------------------------------

function main() {
    var w = new Window("dialog", "jg4d vitrine prep");
    w.orientation = "column";
    w.alignChildren = "fill";
    w.add("statictext", undefined, "Photoshop companion to vitrine_prep.py");
    var g = w.add("group");
    var bBuild = g.add("button", undefined, "Build PSD from _albedo.png");
    var bExp = g.add("button", undefined, "Export PNGs from open PSD");
    var bCancel = w.add("button", undefined, "Cancel");
    var choice = null;
    bBuild.onClick = function () { choice = "build"; w.close(); };
    bExp.onClick = function () { choice = "export"; w.close(); };
    bCancel.onClick = function () { choice = null; w.close(); };
    w.show();

    if (choice === "build") build();
    else if (choice === "export") exportAll();
}

var savedDialogs = app.displayDialogs;
app.displayDialogs = DialogModes.NO;
try {
    main();
} catch (e) {
    alert("jg4d vitrine prep: " + e + (e.line ? "\nline " + e.line : ""));
} finally {
    app.displayDialogs = savedDialogs;
}
