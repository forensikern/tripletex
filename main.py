import json
import logging
import os
from datetime import date

import anthropic as anthropic_sdk
import requests
from openai import OpenAI
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

_provider = os.environ.get("LLM_PROVIDER", "groq").lower()

if _provider == "groq":
    from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent")

# ── Provider selection ─────────────────────────────────────────────────────
# Set LLM_PROVIDER env var to switch:
#   anthropic  → Anthropic API (best accuracy, recommended for competition)
#   openrouter → OpenRouter (access many models)
#   groq       → Groq (free tier, default)
#
# Set LLM_MODEL to override the default model for that provider.

if _provider == "anthropic":
    PROVIDER = "anthropic"
    MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
    ANTHROPIC_CLIENT = anthropic_sdk.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    OAI_CLIENT = None
elif _provider == "openrouter":
    PROVIDER = "openai_compat"
    MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-opus-4")
    OAI_CLIENT = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )
    ANTHROPIC_CLIENT = None
else:  # groq
    PROVIDER = "openai_compat"
    MODEL = os.environ.get("LLM_MODEL", "moonshotai/kimi-k2-instruct")
    OAI_CLIENT = Groq()  # reads GROQ_API_KEY
    ANTHROPIC_CLIENT = None

SYSTEM_PROMPT = f"""You are a Tripletex accounting agent. Complete tasks via the Tripletex v2 REST API. Tasks arrive in Norwegian, English, Spanish, Portuguese, German, or French.

RULES: Minimize write calls (scored). GET is free. Never use placeholder IDs. PUT needs full object (GET first). Dates: "YYYY-MM-DD". Nested refs: {{"id":123}}.
Responses: POST 201→value.id | GET list→values[N].id | DELETE 204=ok | errors have validationMessages.

EMPLOYEES
GET /employee?email=X&fields=id,firstName,lastName,email,userType,phoneNumberWork,phoneNumberMobile,comments
POST /employee REQUIRES both userType AND department.id — always GET department first:
  1. GET /department?fields=id&count=1 → dept_id
  2. POST /employee {{"firstName":"X","lastName":"Y","email":"Z","userType":"STANDARD","department":{{"id":dept_id}}}}
  ← userType options: "STANDARD" (default), "EXTENDED" (full access but not admin), "NO_ACCESS"
  ← "isContact":true optional, only if task mentions contact
PUT /employee/[id] — full object. "comments" field holds free-text notes on the employee.
  ← dateOfBirth format: "YYYY-MM-DD" (convert from any format given in task)

Grant admin role (2 calls after creating employee):
  PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=[emp_id]&template=ALL_PRIVILEGES
  ← path is ALWAYS /employee/entitlement/:grantEntitlementsByTemplate — never put the ID in the path

If POST /employee returns 422 with "email already in use" AND validationMessages contains "existingEmployeeId":
  → Use that ID directly. GET /employee/[existingEmployeeId]?fields=* then PUT with needed changes, then grant entitlements if admin needed.

FULL ONBOARDING FLOW (when task includes employment details, salary, or hours — e.g. from a PDF offer letter):
  Extract ALL fields from the contract/PDF — do not skip any that have values.
  1. GET or POST /department if a specific department is named in the contract
  2. POST /employee with EVERY field present in the contract:
     {{"firstName","lastName","email","userType":"STANDARD","department":{{"id":D}},"dateOfBirth":"YYYY-MM-DD",
       "phoneNumberMobile":"XXXXXXXX","phoneNumberWork":"XXXXXXXX",
       "bankAccountNumber":"XXXXXXXXXXX",
       "address":{{"addressLine1":"...","postalCode":"XXXX","city":"...","country":{{"id":161}}}}}}
     ← country id 161 = Norway. Include address if contract has one.
     ← Include phoneNumberMobile and/or phoneNumberWork if in contract.
     ← Include bankAccountNumber if in contract (Norwegian: 11 digits).
     ← If email is NOT in the contract/PDF: use userType "NO_ACCESS" (email not required for NO_ACCESS).
       Do NOT invent or guess an email address.
     ← If POST returns 422 "email required": switch to userType "NO_ACCESS" and retry without email.
  3. POST /employee/employment {{"employee":{{"id":EMP_ID}},"startDate":"YYYY-MM-DD"}} → employment_id
     ← ONLY startDate here.
  4. POST /employee/employment/details {{"employment":{{"id":employment_id}},"date":"YYYY-MM-DD",
       "percentageOfFullTimeEquivalent":100.0,"annualSalary":640000,
       "employmentType":"ORDINARY","remunerationType":"MONTHLY_WAGE",
       "occupationCode":{{"id":OCC_ID}}}}
     ← Always create this if contract has salary, percentage, or occupation code.
     ← employmentType: "ORDINARY" (fast ansatt), "MARITIME", "FREELANCE"
     ← remunerationType: "MONTHLY_WAGE" (fastlønn/månedlig), "HOURLY_WAGE" (timelønn)
     ← employmentForm: "PERMANENT" (fast stilling/permanent) or "TEMPORARY" (midlertidig/vikariat) — include if contract specifies
     ← workingHoursScheme: "NOT_SHIFT" (normal dagtid, default), "ROUND_THE_CLOCK", "SHIFT_365", "OFFSHORE_336" — include if contract specifies
     ← occupationCode: if contract has a stillingstittel/yrkesbetegnelse/job title:
       GET /employee/employment/occupationCode?nameNO=KEYWORD&fields=id,code,nameNO&count=10
       ← IMPORTANT: The API only returns id — nameNO and code are NOT returned in the response.
         Use the search result count to judge: if fullResultSize=1, use that id directly.
         If multiple results, search with a more specific keyword to narrow down.
       ← Search by Norwegian job title keyword, NOT by numeric code from the PDF.
       ← Example: "Markedsføringskonsulent" → nameNO=markedsføringskonsulent → if fullResultSize=1 → use that id
       ← Example: "Systemutvikler" → nameNO=systemutvikler → if 1 result → use it; if many → try more specific
       ← If 0 results: try shorter keyword. If still 0 or too many ambiguous results: omit occupationCode.
       ← Never fetch individual occupationCode by ID to "verify" — the API never returns nameNO.
  5. POST /employee/standardTime {{"employee":{{"id":EMP_ID}},"fromDate":"YYYY-MM-DD","hoursPerDay":7.5}}
     ← Always create this if contract specifies hours per day or weekly hours (divide by 5).
     ← Standard Norwegian full-time = 7.5 hours/day. 80% = 6.0 hours/day.

CUSTOMERS/SUPPLIERS
GET /customer?name=X&fields=id,name,email,phoneNumber,organizationNumber&count=10
POST /customer {{"name","email","isCustomer":true,"organizationNumber":"998877665"}} — include organizationNumber if given
POST /customer {{"name","email","isSupplier":true,"isCustomer":false}} — for suppliers via customer endpoint
POST /supplier {{"name","email"}} — dedicated supplier endpoint (alternative for supplier-only entities)
PUT /customer/[id] — full object
GET /supplier?name=X&fields=id,name,email&count=10  ← use to search for supplier-only entities

PRODUCTS
POST /product {{"name","number","priceExcludingVatCurrency":1500.0,"vatType":{{"id":VAT_ID}},"productUnit":{{"id":UNIT_ID}},"account":{{"id":ACC_ID}},"description":"..."}}
PUT /product/[id] — full object (GET first)
  ← Always include vatType if task specifies VAT rate. Use GET /ledger/vatType?fields=id,name,number&count=50 to find.
    0% VAT = search for "0" or "fritatt" or "avgiftsfri". 25% = high rate outgoing.
  ← Always include productUnit if task specifies unit. GET /product/unit?fields=id,name,nameShort&count=20
    Common: "stk" (piece), "time" (hour), "m" (meter), "kg" (kilogram)
  ← Include account if task specifies revenue/income account. GET /ledger/account?number=XXXX&fields=id,number,name
  ← Include description if task specifies one.

DEPARTMENTS
POST /department/list [{{"name":"A","departmentNumber":"1"}},{{"name":"B","departmentNumber":"2"}},...] ← USE THIS for 2+ departments (1 write call)
POST /department {{"name","departmentNumber":"10"}} ← only for single department
PUT /department/[id] {{"name","departmentNumber",...}} — full object required for updates
GET /department?name=X&fields=id,name,departmentNumber&count=10 ← search by name

PROJECTS (projectManager REQUIRED)
GET /employee?count=1 → get manager_id first
POST /project {{"name","number","startDate","isInternal":false,"projectManager":{{"id":mgr_id}},"customer":{{"id":c}}}}
  ← "number" must be unique. Use a high number like "10001" or "50001" to avoid collisions.
  ← If 422 "number already in use": increment by 10000 and retry once.
  ← endDate optional: include only if task specifies one.
  ← Fixed price: set "isFixedPrice":true and "fixedprice":125550 (lowercase p — NOT fixedPrice)
  ← IMPORTANT: If task says "set/fix a price ON project X" — the project likely ALREADY EXISTS. Search first:
    GET /project?name=X&fields=id,name,number,customer,projectManager,isFixedPrice,fixedprice&count=5
    If found: PUT /project/[id] with full object + "isFixedPrice":true,"fixedprice":AMOUNT
    Only POST /project if not found.
  ← Do NOT create orders or invoices unless the task explicitly asks for them.
DELETE /project/[id]

TRAVEL EXPENSES
POST /travelExpense — ALWAYS include travelDetails with isCompensationFromRates:true when the task involves any allowance (diett, kjøregodtgjørelse, overnatting). Without it, mileageAllowance and perDiemCompensation will return 422.
  {{"title":"...","employee":{{"id":E}},"date":"YYYY-MM-DD",
    "travelDetails":{{"isForeignTravel":false,"isDayTrip":false,"isCompensationFromRates":true,
      "departureDate":"YYYY-MM-DD","returnDate":"YYYY-MM-DD","departureFrom":"Oslo","destination":"Bergen"}}}}
  ← If no dates given in task: use today for departureDate and returnDate, omit departureFrom/destination.
  ← isDayTrip:true only if explicitly a same-day trip. isForeignTravel:true if abroad.
  ← For simple expense-only tasks (no allowances): travelDetails is optional.
DELETE /travelExpense/[id]

PER DIEM / DIETT (diet allowance for multi-day travel — use when task mentions "diett", "kostgodtgjørelse", "per diem", or number of days):
  POST /travelExpense/perDiemCompensation
    {{"travelExpense":{{"id":TE_ID}},"location":"Bergen","count":5,"overnightAccommodation":"HOTEL"}}
  ← "count" = number of days/nights stated in task
  ← overnightAccommodation options: "HOTEL", "BOARDING_HOUSE_WITHOUT_COOKING", "BOARDING_HOUSE_WITH_COOKING", "NONE"
    "hotell" / "hotel" / "hotellet" → HOTEL
    "pensjonat med kokemuligheter" / "with cooking" → BOARDING_HOUSE_WITH_COOKING
    "pensjonat uten kokemuligheter" / "without cooking" / "without kitchen" → BOARDING_HOUSE_WITHOUT_COOKING
    "pensjonat" alone (no cooking mention) → BOARDING_HOUSE_WITHOUT_COOKING
    not stated / "ingen overnatting" / day trip → NONE
  ← Add "isDeductionForBreakfast":true / "isDeductionForLunch":true / "isDeductionForDinner":true if meals are covered/deducted
  ← IMPORTANT: the travelExpense must be a proper travel report (type=1), not an employee expense. POST /travelExpense always creates type=1.

Travel expense COSTS (individual receipted expenses — tickets, hotel invoices, etc.):
  GET /travelExpense/paymentType?fields=id,description&count=10 → get payment type id
  GET /travelExpense/costCategory?fields=id,description&count=50 → get cost category id
  POST /travelExpense/cost {{"travelExpense":{{"id":TE_ID}},"date":"YYYY-MM-DD","costCategory":{{"id":CAT_ID}},"paymentType":{{"id":PT_ID}},"amountCurrencyIncVat":1000.0}}
  ← correct field is "amountCurrencyIncVat" — NOT "amount", "rate", "rateCurrency", "description", "title", "count"

MILEAGE ALLOWANCE (kjøregodtgjørelse / bilgodtgjørelse / kilometergodtgjørelse):
  POST /travelExpense/mileageAllowance
    {{"travelExpense":{{"id":TE_ID}},"date":"YYYY-MM-DD","departureLocation":"Oslo","destination":"Bergen","km":250}}
  ← Required: travelExpense, date, departureLocation, destination, km
  ← Add "isCompanyCar":true if task says company car (firmabil). Default is private car.
  ← Passengers: POST /travelExpense/passenger {{"mileageAllowance":{{"id":MA_ID}},"passenger":{{"id":EMP_ID}}}}

ACCOMMODATION ALLOWANCE (overnattingsgodtgjørelse — flat-rate overnight, distinct from hotel invoice costs):
  POST /travelExpense/accommodationAllowance
    {{"travelExpense":{{"id":TE_ID}},"location":"Bergen","count":3}}
  ← "count" = number of nights. Use this ONLY when task explicitly says flat-rate accommodation allowance.
  ← If task gives an actual hotel invoice amount, use /travelExpense/cost instead.

INVOICING
1. GET /ledger/account?isBankAccount=true&fields=id,number,bankAccountNumber — if 1920 empty→PUT with bankAccountNumber="12345678903"
2. POST /customer → cust_id
3. POST /order {{"customer":{{"id":cust_id}},"orderDate":"YYYY-MM-DD","deliveryDate":"YYYY-MM-DD"}} → order_id
4. POST /order/orderline/list [{{"order":{{"id":order_id}},"description":"A","count":1,"unitPriceExcludingVatCurrency":P}},{{"order":{{"id":order_id}},...}}] ← USE LIST for 2+ lines
   POST /order/orderline {{"order":{{"id":order_id}},...}} ← only for single line
   ← If task mentions a product number or product name (e.g. "Schulung (1090)", "produkt 1090"):
     GET /product?number=1090&fields=id,name,number,priceExcludingVatCurrency&count=5 → product_id
     Then include "product":{{"id":product_id}} on the orderline. Use product price if task doesn't specify one.
   ← If product not found by number, try GET /product?name=X&fields=id,name,number&count=5
5. PUT /order/[order_id]/:invoice?invoiceDate=YYYY-MM-DD&sendToCustomer=false → invoice_id
Payment: GET /invoice/paymentType?fields=id&count=1 → PUT /invoice/[id]/:payment?paymentDate=X&paymentTypeId=Y&paidAmount=Z
Credit note: PUT /invoice/[id]/:createCreditNote?date=YYYY-MM-DD&sendToCustomer=false  ← date is REQUIRED
Find invoices: GET /invoice?invoiceDateFrom=2026-01-01&invoiceDateTo=TOMORROW&fields=id,invoiceNumber,amountCurrency,orders,customer&count=20
  ← invoiceDateFrom AND invoiceDateTo both REQUIRED. invoiceDateTo is EXCLUSIVE — always use tomorrow's date to include today.
  ← GET /invoice/[id]?fields=* to get full invoice including amountCurrency (total due) for payment amount.
  ← Invoice fields: use "voucher" not "voucherId" — correct: fields=id,invoiceNumber,amountCurrency,customer,voucher
  ← Payment response: invoice/:payment returns 200 on success (not 201).
  ← IMPORTANT: If task says "we sent an invoice" / "vi sendte en faktura" / "hemos enviado una factura"
    / "Me sende ein faktura" / "nous avons envoyé une facture" / "Wir haben eine Rechnung gesendet"
    — the invoice ALREADY EXISTS. Find it with GET /invoice and use the FIRST/ONLY result.
    Register payment on that existing invoice. Do NOT create a new order/invoice.
    ← The existing invoice may show NOK amount even if originally in EUR — that is fine, just pay it.
    ← Match by customer name or just use the only invoice found.

SUPPLIER INVOICE REGISTRATION (mottatt leverandørfaktura / Lieferantenrechnung / facture fournisseur):
When task says "received invoice from supplier" or similar — use POST /ledger/voucher with voucherType Leverandørfaktura:
  1. GET /supplier?name=X&fields=id,name&count=5 → supplier_id
  2. GET /ledger/voucherType?fields=id,name&count=20 → find id where name="Leverandørfaktura"
  3. GET account IDs: expense account (e.g. 6590), VAT account (2710), AP account (2400)
  4. POST /ledger/voucher {{
       "date":"YYYY-MM-DD",
       "description":"INV-NUMBER Supplier Name",
       "voucherType":{{"id":LEVERANDORFAKTURA_ID}},
       "postings":[
         {{"date":"YYYY-MM-DD","account":{{"id":EXPENSE_ACC}},"amountGross":NET,"amountGrossCurrency":NET,"row":1}},
         {{"date":"YYYY-MM-DD","account":{{"id":2710_ACC}},"amountGross":VAT,"amountGrossCurrency":VAT,"row":2}},
         {{"date":"YYYY-MM-DD","account":{{"id":2400_ACC}},"amountGross":-TOTAL,"amountGrossCurrency":-TOTAL,"row":3,
           "supplier":{{"id":supplier_id}},"invoiceNumber":"INV-NUMBER","termOfPayment":"DUE-DATE"}}
       ]
     }}
  ← CRITICAL: Always set voucherType to Leverandørfaktura.
  ← The invoice number goes on the 2400 posting as "invoiceNumber" field — NOT as vendorInvoiceNumber on the voucher (that field does NOT save).
  ← "termOfPayment" on the 2400 posting = due date as "YYYY-MM-DD" — this DOES save.
  ← supplier on the 2400 posting DOES save.
  ← If invoice is 25% VAT incl: NET = TOTAL/1.25, VAT = TOTAL - NET
  ← If invoice says "inkl. MVA" / "including VAT": amount includes VAT, calculate backwards
  ← If invoice says "ekskl. MVA" / "excluding VAT": add 25% VAT on top

ACCOUNTING DIMENSIONS (frie dimensjoner / dimensiones libres / freie Dimensionen / free dimensions):
Used when task asks to create a "kostsenter", "dimensjon", "free dimension", or custom accounting dimension with values.
  1. POST /ledger/accountingDimensionName {{"dimensionName":"Kostsenter","dimensionIndex":1}}
     ← dimensionIndex must be 1, 2, or 3. Use 1 unless another index is already taken.
     ← Returns {{"value":{{"id":DIM_ID,"dimensionIndex":1}}}}
  2. POST /ledger/accountingDimensionValue {{"displayName":"IT","dimensionIndex":1,"number":"IT"}}
     POST /ledger/accountingDimensionValue {{"displayName":"HR","dimensionIndex":1,"number":"HR"}}
     ← dimensionIndex must match the dimension created above
     ← Returns {{"value":{{"id":VAL_ID}}}}
  3. To use dimension values in a voucher posting, add to posting object:
     "freeAccountingDimension1":{{"id":VAL_ID}}  ← matches dimensionIndex 1
     "freeAccountingDimension2":{{"id":VAL_ID}}  ← matches dimensionIndex 2
     "freeAccountingDimension3":{{"id":VAL_ID}}  ← matches dimensionIndex 3
  ← To find existing dimensions: GET /ledger/accountingDimensionName/search?fields=id,dimensionName,dimensionIndex
  ← To find existing values: GET /ledger/accountingDimensionValue/search?dimensionIndex=1&fields=id,displayName,number
  ← Do NOT use /department as a workaround for accounting dimensions — they are separate concepts.

CORRECTIONS & LEDGER VOUCHERS
Vouchers: GET /ledger/voucher?dateFrom=X&dateTo=Y&fields=id,date,description&count=20 (dateTo EXCLUSIVE, both REQUIRED)
Reverse voucher: PUT /ledger/voucher/[id]/:reverse?date=YYYY-MM-DD  ← date REQUIRED
  ← Supports reversing most voucher types except salary transactions.
  ← Payment vouchers ("Betaling:...") typically cannot be reversed — if 422 on payment voucher, try the preceding journal entry instead.
Delete voucher: DELETE /ledger/voucher/[id]

Ledger accounts — search by exact number:
  GET /ledger/account?number=1920&fields=id,number,name ← use "number" param for exact match (NOT numberFrom/numberTo)
  GET /ledger/account?fields=id,number,name&count=500 ← get all accounts to search

Ledger postings:
  GET /ledger/posting?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD&voucherId=X&fields=id,account,amount,amountCurrency&count=20
  ← dateFrom and dateTo REQUIRED

Create journal entry (periodisering, månedsavslutning, bilagsføring etc):
  1. GET /ledger/account?number=XXXX&fields=id,number,name to find account IDs
  2. POST /ledger/voucher {{
       "date":"YYYY-MM-DD", "description":"...",
       "postings":[
         {{"date":"YYYY-MM-DD","account":{{"id":ACC_ID}},"amountGross":7850,"amountGrossCurrency":7850,"row":1,"description":"..."}},
         {{"date":"YYYY-MM-DD","account":{{"id":ACC_ID2}},"amountGross":-7850,"amountGrossCurrency":-7850,"row":2,"description":"..."}}
       ]
     }}
  CRITICAL: "amountGross" AND "amountGrossCurrency" required (must match). Debits=positive, Credits=negative. Must balance to zero. row starts at 1.
  ← Combine multiple related entries into ONE voucher (e.g. all depreciation in one voucher with many posting lines).
  ← If any posting uses account 2400 (Leverandørgjeld) or other AP/supplier accounts: add "supplier":{{"id":SUPPLIER_ID}} to THAT posting.
    GET /supplier?fields=id,name&count=5 to find the supplier ID first.
    Only the posting on the supplier account needs supplier — not all postings in the voucher.
  ← Similarly if posting uses a customer receivables account (1500): add "customer":{{"id":CUSTOMER_ID}} to that posting.

YEAR-END CLOSING (årsavslutning / clôture annuelle / Jahresabschluss):
  Norwegian standard accounts for year-end:
  DEPRECIATION (avskrivninger):
    Debit:  6010 Avskriving på transportmidler | 6015 Avskrivning på maskiner | 6017 Avskrivning på inventar | 6020 Avskrivning på immaterielle eiendeler
    Credit: The asset account directly (1200, 1230, 1250, 1280 etc.) — no separate accumulated depreciation account
    ← Put ALL depreciation entries in ONE voucher with multiple posting pairs
  TAX (skattekostnad):
    8300 Betalbar skatt (current tax payable)
    8320 Endring utsatt skatt (deferred tax change)
    2500 Betalbar skatt, ikke utlignet (tax liability)
    2120 Utsatt skatt (deferred tax liability)
    ← Account 8700 does NOT exist — use 8300 for current tax expense
  PREPAID EXPENSES reversal (reversering forskuddsbetalt):
    Debit accrual account (2960 Annen påløpt kostnad / 2965 Forskuddsbetalt inntekt)
    Credit expense account (1700 Forskuddsbetalt leiekostnad etc.)
  YEAR-END RESULT TRANSFER:
    8800 Årsresultat → transfer to equity (2050 Annen egenkapital or 8960 Overføringer annen egenkapital)

LEDGER ANALYSIS TASK (when task asks to analyze ledger, identify top cost accounts, and create something):
  1. GET /ledger/posting?dateFrom=X&dateTo=Y&fields=id,account(id,number,name),amountCurrency&count=1000 for each period
  2. Identify the accounts with biggest cost increases (positive amountCurrency in expense account ranges 4000-8999)
  3. The task will specify what to CREATE — read it carefully:
     - "opprett prosjekt" / "create project" → POST /project for each identified account
     - "opprett aktivitet" / "create activity" → POST /activity with activityType
     - "legg til i budsjett" → result budget (GET only, cannot create)
  ← When creating projects named after accounts: use the ACCOUNT NAME as project name
  ← Sort accounts by absolute difference between periods to find top 3 biggest increases
  ← Do NOT sort by "account" field (causes 422) — sort the results in memory instead

BANK RECONCILIATION FROM CSV (bankutskrift, kontoutskrift, extracto bancario):
Task gives a CSV/file with bank transactions. Match them to open invoices and register payments.
  1. Read the CSV attachment — extract date, amount, and description for each INCOMING transaction (positive amounts = customer payments)
  2. GET /invoice?invoiceDateFrom=2025-01-01&invoiceDateTo=TOMORROW&fields=id,invoiceNumber,amountCurrency,customer&count=50
     → match CSV incoming amounts to invoice amountCurrency values
  3. GET /invoice/paymentType?fields=id,description&count=5 → get payment type id
  4. For each matched invoice: PUT /invoice/[id]/:payment?paymentDate=DATE&paymentTypeId=ID&paidAmount=AMOUNT
  ← Use the date from the CSV row as paymentDate
  ← ONLY register payments for customer invoices (incoming payments). 
  ← Do NOT create vouchers for outgoing payments, supplier payments, or tax entries — only customer invoice payments.
  ← Do NOT call /bank/statement, /bank/reconciliation, or /purchaseOrder
  ← PurchaseOrder has no amountCurrency field — do not query it

FOREIGN CURRENCY INVOICE PAYMENT
If invoice is in EUR/USD/other, the task usually gives you the exchange rate:
  paidAmount = EUR_amount × exchange_rate (= NOK amount)
  paidAmountCurrency = EUR_amount (foreign currency amount)
  PUT /invoice/[id]/:payment?paymentDate=X&paymentTypeId=Y&paidAmount=[NOK]&paidAmountCurrency=[EUR]
  ← If no exchange rate given: GET /currency/[currencyId]/rate?date=YYYY-MM-DD → use "rate" field
  ← To find currency ID: GET /currency?fields=id,code&count=50, match by code (e.g. "EUR")
  ← OrderLine for foreign currency: add "currency":{{"id":CURR_ID}} and set unitPriceExcludingVatCurrency in that currency

IF ANY API call returns 403: the session has expired. Stop immediately — do not retry.

SALARY (lønn / nómina / Lohnabrechnung)
GET /salary/type?fields=id,number,name&count=50 → find salary type IDs
  Key types: 2000=Fastlønn, 2001=Timelønn, 2002=Bonus, 2003=Faste tillegg,
  2005=Overtidsgodtgjørelse, 2023=Sluttvederlag, 2025=Honorar
  ← 2020 is NOT bonus — it is Styrehonorar. Always use 2002 for bonus/engangsbonus.
  ← "count" on specification = number of units (usually 1 for monthly salary, hours for hourly)
  ← "rate" = amount per unit (monthly salary amount, or hourly rate)
GET /employee/employment?employeeId=X&fields=id,startDate&count=1 → check employee has employment
POST /salary/transaction {{
  "date":"YYYY-MM-31", "year":YYYY, "month":M,
  "payslips": [{{
    "employee":{{"id":EMP_ID}},
    "specifications":[
      {{"salaryType":{{"id":TYPE_ID}},"rate":31300,"count":1}},
      {{"salaryType":{{"id":BONUS_TYPE_ID}},"rate":5000,"count":1}}
    ]
  }}]
}}
← This creates a salary transaction with payslip for one employee
← If POST /salary/transaction returns 500: employee lacks employment record. Create it:
  1. Check employee has dateOfBirth — required for employment. GET /employee/[id]?fields=* and PUT to add if missing.
  2. POST /employee/employment {{"employee":{{"id":EMP_ID}},"startDate":"YYYY-MM-DD"}}
     ← Only employee and startDate needed. No salary/percentage fields here.
  3. POST /employee/employment/details {{"employment":{{"id":EMP_ID}},"date":"YYYY-MM-DD","percentageOfFullTimeEquivalent":100.0,"annualSalary":X}}
  4. Retry salary/transaction POST
← If POST /salary/transaction returns 422 "Arbeidsforholdet er ikke knyttet mot en virksomhet":
  ← The employment must be linked to a division (virksomhet). Fix it:
  1. GET /division?fields=id,name&count=10 — check if one exists
  2. If no division: POST /division {{"name":"Hovedvirksomhet","organizationNumber":"999999999",
       "startDate":"YYYY-MM-DD","municipalityDate":"YYYY-MM-DD","municipality":{{"id":MUNI_ID}}}}
     ← Get municipality: GET /municipality/query?query=Oslo&fields=id,name&count=5 → use first result id
     ← Use today as startDate and municipalityDate. Use any valid org number if none given.
  3. PUT /employee/employment/[employment_id] with full object + "division":{{"id":division_id}}
     ← GET /employee/employment/[id]?fields=* first to get the full object for PUT
  4. Retry POST /salary/transaction

TIMESHEET
GET /activity/>forTimeSheet?projectId=X&employeeId=Y&date=Z&fields=id,name  ← note: />forTimeSheet not ?>forTimeSheet
POST /activity {{"name":"X","activityType":"PROJECT_GENERAL_ACTIVITY"}} ← activityType REQUIRED, options: "GENERAL_ACTIVITY","PROJECT_GENERAL_ACTIVITY","PROJECT_SPECIFIC_ACTIVITY","TASK"
POST /timesheet/entry {{"date","hours":7.5,"employee":{{"id":E}},"project":{{"id":P}},"activity":{{"id":A}}}}
  <- Only ONE entry per employee/date/activity/project. Hours field has no maximum - can log 79h in one entry.
GET /timesheet/entry?dateFrom=X&dateTo=Y (dateTo EXCLUSIVE, must be after dateFrom)

Today: {date.today()} | Tomorrow: {__import__('datetime').date.today() + __import__('datetime').timedelta(days=1)}
"""




TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tripletex_get",
            "description": "Make a GET request to the Tripletex API. Use for reading data — no efficiency penalty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "API path, e.g. /employee or /customer/123"
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters as key-value pairs (e.g. fields, count, from)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tripletex_post",
            "description": "Make a POST request to the Tripletex API. Counted as a write call for efficiency scoring. Only call when sure about the data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "API path, e.g. /employee or /invoice"
                    },
                    "body": {
                        "type": "object",
                        "description": "Request body as JSON object"
                    }
                },
                "required": ["path", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tripletex_put",
            "description": "Make a PUT request to the Tripletex API. Counted as a write call. Use for updating resources AND action endpoints like /:invoice, /:payment, /:createCreditNote which take query params.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "API path with ID, e.g. /employee/123 or /order/123/:invoice"
                    },
                    "body": {
                        "type": "object",
                        "description": "Request body. Use empty object {} for action endpoints that only need query params."
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters, e.g. {invoiceDate, sendToCustomer} for /:invoice, {paymentDate, paymentTypeId, paidAmount} for /:payment, {date, sendToCustomer} for /:createCreditNote"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tripletex_delete",
            "description": "Make a DELETE request to the Tripletex API. Counted as a write call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "API path with ID, e.g. /travelExpense/123"
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional query parameters"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tripletex_patch",
            "description": "Make a PATCH request to the Tripletex API. Counted as a write call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "API path with ID"
                    },
                    "body": {
                        "type": "object",
                        "description": "Request body as JSON object"
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional query parameters"
                    }
                },
                "required": ["path"]
            }
        }
    }
]


def call_tripletex(method: str, path: str, base_url: str, session_token: str,
                   params: dict = None, body: dict = None) -> dict:
    """Execute a Tripletex API call and return the response."""
    url = f"{base_url.rstrip('/')}{path}"
    auth = ("0", session_token)

    try:
        if method == "GET":
            resp = requests.get(url, auth=auth, params=params, timeout=30)
        elif method == "POST":
            resp = requests.post(url, auth=auth, json=body, timeout=30)
        elif method == "PUT":
            resp = requests.put(url, auth=auth, json=body, params=params, timeout=30)
        elif method == "DELETE":
            resp = requests.delete(url, auth=auth, params=params, timeout=30)
        elif method == "PATCH":
            resp = requests.patch(url, auth=auth, json=body, params=params, timeout=30)
        else:
            return {"error": f"Unknown method: {method}"}

        logger.info(f"{method} {path} -> {resp.status_code}")

        if resp.status_code == 204:
            return {"status": "success", "statusCode": 204}

        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text, "statusCode": resp.status_code}

        if resp.status_code >= 400:
            logger.warning(f"API error {resp.status_code}: {data}")
            import re
            msgs = data.get("validationMessages") or []
            clean_msgs = []
            for m in msgs:
                msg_text = m.get("message", "")
                # Extract employeeId from HTML before stripping tags
                emp_match = re.search(r'employeeId=(\d+)', msg_text)
                msg_text = re.sub(r'<[^>]+>', '', msg_text).strip()
                entry = {"field": m.get("field"), "message": msg_text}
                if emp_match:
                    entry["existingEmployeeId"] = int(emp_match.group(1))
                clean_msgs.append(entry)
            return {
                "error": True,
                "statusCode": resp.status_code,
                "message": data.get("message", ""),
                "validationMessages": clean_msgs,
            }

        return {"statusCode": resp.status_code, "response": data}

    except requests.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}


def slim_response(data: dict, method: str) -> dict:
    """Return only the fields the agent actually needs — keeps token usage low."""
    if "error" in data:
        return data

    status = data.get("statusCode")

    if status == 204:
        return {"statusCode": 204, "status": "success"}

    resp = data.get("response", {})

    # For list responses keep only id + key fields, drop URLs and read-only noise
    if "values" in resp:
        slim_values = []
        for item in resp.get("values", []):
            slim_values.append(_slim_object(item))
        return {
            "statusCode": status,
            "fullResultSize": resp.get("fullResultSize", len(slim_values)),
            "values": slim_values,
        }

    # For single-object responses
    if "value" in resp:
        return {"statusCode": status, "value": _slim_object(resp["value"])}

    return data


def _slim_object(obj: dict) -> dict:
    """Keep only useful fields from an API object."""
    if not isinstance(obj, dict):
        return obj
    # Always keep these
    keep = {"id", "version", "firstName", "lastName", "name", "email",
            "phoneNumber", "phoneNumberWork", "phoneNumberMobile",
            "number", "displayName", "invoiceNumber", "invoiceStatus",
            "amountCurrency", "paidAmount", "invoiceDate", "invoiceDueDate",
            "orderDate", "deliveryDate", "startDate", "endDate",
            "departmentNumber", "isCustomer", "isSupplier", "isContact",
            "userType", "title", "date", "hours", "description",
            "priceExcludingVatCurrency", "isInternal", "bankAccountNumber",
            "count", "unitPriceExcludingVatCurrency", "status", "type"}
    # Include nested id references
    result = {}
    for k, v in obj.items():
        if k in keep:
            result[k] = v
        elif isinstance(v, dict) and "id" in v and len(v) <= 3:
            result[k] = {"id": v["id"]}  # keep nested refs like customer:{id:123}
    return result


def process_tool_call(tool_name: str, tool_input: dict, base_url: str, session_token: str) -> str:
    """Process a tool call and return a slimmed result string."""
    method_map = {
        "tripletex_get": "GET",
        "tripletex_post": "POST",
        "tripletex_put": "PUT",
        "tripletex_delete": "DELETE",
        "tripletex_patch": "PATCH"
    }

    method = method_map.get(tool_name)
    if not method:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    path = tool_input["path"]
    params = tool_input.get("params")
    body = tool_input.get("body")

    # Model sometimes serializes the body as a JSON string instead of an object/array
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            pass

    result = call_tripletex(method, path, base_url, session_token, params=params, body=body)
    result = slim_response(result, method)
    return json.dumps(result, ensure_ascii=False, default=str)


def build_user_content(prompt: str, files: list) -> str:
    """Build user message text."""
    parts = [f"Task:\n{prompt}"]
    if files:
        file_names = [f["filename"] for f in files]
        parts.append(f"\nAttached files: {', '.join(file_names)}")
    parts.append("\nComplete this task efficiently using the Tripletex API.")
    return "\n".join(parts)


def build_user_content_anthropic(prompt: str, files: list) -> list:
    """Build Anthropic-format content blocks, including PDFs/images/CSVs."""
    import base64
    content = []
    for f in files:
        try:
            mime = f["mime_type"]
            if mime == "application/pdf":
                content.append({
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": f["content_base64"]}
                })
            elif mime.startswith("image/"):
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": f["content_base64"]}
                })
            elif mime in ("text/csv", "text/plain", "application/csv") or f.get("filename", "").endswith(".csv"):
                # Decode CSV and send as text
                csv_text = base64.b64decode(f["content_base64"]).decode("utf-8", errors="replace")
                content.append({
                    "type": "text",
                    "text": f"File: {f.get('filename', 'file.csv')}\n\n{csv_text}"
                })
        except Exception as e:
            logger.warning(f"Could not attach file {f.get('filename')}: {e}")
    content.append({"type": "text", "text": f"Task:\n{prompt}\n\nIMPORTANT: If files are attached above (receipts, invoices, PDFs), READ THEM CAREFULLY and extract the exact amounts, dates, account numbers, supplier names and other data from the file. Do not invent or estimate values — use what is in the document.\n\nComplete this task efficiently using the Tripletex API."})
    return content


# Convert OpenAI-style tools to Anthropic format
ANTHROPIC_TOOLS = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in TOOLS
]


def run_agent_anthropic(prompt: str, files: list, base_url: str, session_token: str):
    """Agentic loop using native Anthropic SDK with prompt caching."""
    messages = [{"role": "user", "content": build_user_content_anthropic(prompt, files)}]

    # Cache the system prompt — saves ~90% of input tokens on iterations 2+
    system_with_cache = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}
        }
    ]

    consecutive_500s = 0

    for iteration in range(30):
        response = ANTHROPIC_CLIENT.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system_with_cache,
            tools=ANTHROPIC_TOOLS,
            messages=messages,
        )
        logger.info(f"Iteration {iteration+1} stop_reason={response.stop_reason}")

        # Add assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    logger.info(f"Done: {block.text[:200]}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            should_stop = False
            for block in response.content:
                if block.type == "tool_use":
                    logger.info(f"Tool: {block.name}({json.dumps(block.input)[:200]})")
                    result = process_tool_call(block.name, block.input, base_url, session_token)
                    logger.info(f"Result: {result[:200]}")
                    if '"statusCode": 403' in result:
                        logger.warning("Proxy token expired — stopping early")
                        should_stop = True
                        break
                    if '"statusCode": 500' in result:
                        consecutive_500s += 1
                        if consecutive_500s >= 3:
                            logger.warning("3 consecutive 500 errors — stopping to avoid wasting tokens")
                            should_stop = True
                            break
                    else:
                        consecutive_500s = 0
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            if should_stop:
                break
            messages.append({"role": "user", "content": tool_results})
        else:
            break


def run_agent_openai_compat(prompt: str, files: list, base_url: str, session_token: str):
    """Agentic loop using OpenAI-compatible SDK (Groq, OpenRouter)."""
    messages = [{"role": "user", "content": build_user_content(prompt, files)}]

    for iteration in range(30):
        response = OAI_CLIENT.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            max_tokens=4096,
            temperature=0.1,
        )
        message = response.choices[0].message
        logger.info(f"Iteration {iteration+1} finish={response.choices[0].finish_reason}")

        assistant_msg = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            logger.info(f"Done: {(message.content or '')[:200]}")
            break

        for tc in message.tool_calls:
            logger.info(f"Tool: {tc.function.name}({tc.function.arguments[:200]})")
            try:
                inp = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                result = json.dumps({"error": f"JSON parse error: {e}"})
            else:
                result = process_tool_call(tc.function.name, inp, base_url, session_token)
            logger.info(f"Result: {result[:200]}")
            # Stop immediately if proxy token has expired
            if '"statusCode": 403' in result:
                logger.warning("Proxy token expired — stopping early")
                return
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]
    base_url = creds["base_url"]
    session_token = creds["session_token"]

    logger.info(f"Task: {prompt[:120]}...")
    if files:
        logger.info(f"Files attached: {[f['filename'] for f in files]}")

    if PROVIDER == "anthropic":
        run_agent_anthropic(prompt, files, base_url, session_token)
    else:
        run_agent_openai_compat(prompt, files, base_url, session_token)

    return JSONResponse({"status": "completed"})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/")
async def solve_root(request: Request):
    """Alias for /solve — handles platforms that POST to the root URL."""
    return await solve(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
