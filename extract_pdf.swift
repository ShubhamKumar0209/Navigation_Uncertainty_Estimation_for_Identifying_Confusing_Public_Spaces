import Foundation
import PDFKit

let args = CommandLine.arguments
if args.count < 2 {
    print("Usage: extract_pdf.swift <path>")
    exit(1)
}

let url = URL(fileURLWithPath: args[1])
guard let document = PDFDocument(url: url) else {
    print("Could not load PDF")
    exit(1)
}

var fullText = ""
for i in 0..<document.pageCount {
    if let page = document.page(at: i), let text = page.string {
        fullText += text + "\n"
    }
}
print(fullText)
